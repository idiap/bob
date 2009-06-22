#include "MultiVariateMAPDiagonalGaussianDistribution.h"

namespace Torch {

MultiVariateMAPDiagonalGaussianDistribution::MultiVariateMAPDiagonalGaussianDistribution(MultiVariateDiagonalGaussianDistribution *prior_)
{
   	CHECK_FATAL(prior_ != NULL); 

	addFOption("map factor", 0.5, "map factor");

	prior = prior_;

	n_inputs = prior->m_parameters->getI("n_inputs");
	n_means = prior->m_parameters->getI("n_means");

	//Torch::print("MultiVariateMAPDiagonalGaussianDistribution()\n");
	//Torch::print("   n_inputs = %d\n", n_inputs);
	//Torch::print("   n_means = %d\n", n_means);
	
	//prior->m_parameters->print();

	m_parameters->copy(prior->m_parameters);

	//m_parameters->print();

	MultiVariateNormalDistribution::resize(n_inputs, n_means);
	resize(n_inputs, n_means);
}

MultiVariateMAPDiagonalGaussianDistribution::~MultiVariateMAPDiagonalGaussianDistribution()
{
}

bool MultiVariateMAPDiagonalGaussianDistribution::prepare()
{
	MultiVariateDiagonalGaussianDistribution::prepare();
	
	map_factor = getFOption("map factor");

	return true;
}

bool MultiVariateMAPDiagonalGaussianDistribution::EMaccPosteriors(const DoubleTensor *input, const double input_posterior)
{
	double *src = (double *) input->dataR();

   	/** Computes the posterior for each gaussian

		\begin{equation}
			P(q_j | x_i) = \frac{P(q_j) \times p(x_i | q_j)}{\sum_{k=1}^K P(q_k) \times p(x_i | q_k)}
		\end{equation}
	*/

   	/** Computes and keeps the numerator and cumulates the denominator
	*/
   	double posterior_denominator;
	if(use_log)
	{
		posterior_denominator = sampleProbability(src);
		for(int j = 0 ; j < n_means ; j++) posterior_numerator[j] = log(weights[j]) + current_likelihood_one_mean[j];
	}
	else
	{
   		posterior_denominator = 0.0;

		for(int j = 0 ; j < n_means ; j++) 
		{
			posterior_numerator[j] = weights[j] * sampleProbabilityOneGaussian(src, j); 
			posterior_denominator += posterior_numerator[j];
		}
	}

	/** Accumulates weights, means and variances weighted by the posterior

		\begin{equation}
			posterior_j = P(q_j | x_i)
			            = posterior_numerator[j] / posterior_denominator

			posterior_numerator[j]  = P(q_j) \times p(x_i | q_j)
						= weights[j] * sampleProbabilityOneGaussian(data[i], j)

			posterior_denominator   = \sum_{j=1}^{n_gaussians} P(q_j) \times p(x_i | q_j)
						= \sum_{j=1}^{n_gaussians} posterior_numerator[j]


			acc_posteriors_weights[j] = \sum_{i=1}^{n_samples} P(q_j | x_i)
						  = \sum_{i=1}^{n_samples} posterior_j

			acc_posteriors_sum_weights = \sum_{j=1}^{n_gaussians} \sum_{i=1}^{n_samples} P(q_j | x_i)
						   = \sum_{j=1}^{n_gaussians} \sum_{i=1}^{n_samples} posterior_j

			acc_posteriors_means[j]	= \sum_{i=1}^{n_samples} P(q_j | x_i) \times x_i

			acc_posteriors_variances[j] = \sum_{i=1}^{n_samples} P(q_j | x_i) \times x_i^2

		\end{equation}
	*/
	for(int j = 0 ; j < n_means ; j++)
	{
	   	double posterior_j;
		if(use_log) posterior_j = exp(input_posterior + posterior_numerator[j] - posterior_denominator);
		else posterior_j = posterior_numerator[j] / posterior_denominator;

		acc_posteriors_weights[j] += posterior_j;
		acc_posteriors_sum_weights += posterior_j;

		for(int k = 0 ; k < n_inputs ; k++) 
		{
			double z = src[k];

			acc_posteriors_means[j][k] += posterior_j * z;

			acc_posteriors_variances[j][k] += posterior_j * z * z;
		}
	}

	return true;
}

bool MultiVariateMAPDiagonalGaussianDistribution::EMupdate()
{
	/*
	real epsilon = 10*REAL_EPSILON;
	real* p_weights_acc = weights_acc;
	if(learn_means)
	for (int i=0;i<n_gaussians;i++,p_weights_acc++) 
	{
		if (*p_weights_acc <= (prior_weights + epsilon))
		{
			real* p_means_prior_i = prior_distribution->means[i];
			real* p_means_i = means[i];
			for (int j=0;j<n_inputs;j++) 
				*p_means_i++ = *p_means_prior_i++;
		}
		else
		{
			real* p_means_prior_i = prior_distribution->means[i];
			real* p_means_i = means[i];
			real* p_means_acc_i = means_acc[i];
			for (int j=0;j<n_inputs;j++)
			{
				*p_means_i++ = (weight_on_prior * *p_means_prior_i++) + ((1 - weight_on_prior) * *p_means_acc_i++ / *p_weights_acc);
			}
		}
	}
	*/

	for(int j = 0 ; j < n_means ; j++)
	{
	   	/** Update rule for weights:
			\begin{equation}
			\lambda_j = \frac{\sum_{i=1}^{n_samples} P(q_j | x_i)}{\sum_{k=1}^{n_gaussians} \sum_{i=1}^{n_samples} P(q_k | x_i)}
			\end{equation}
		*/
		weights[j] = acc_posteriors_weights[j] / acc_posteriors_sum_weights;

		for(int k = 0 ; k < n_inputs ; k++)
		{
	   		/** Update rule for means:
				\begin{equation}
				\mu_j = \frac{\sum_{i=1}^{n_samples} P(q_j | x_i) \times x_i}{\sum_{i=1}^{n_samples} P(q_j | x_i)}
				\end{equation}
			*/
			means[j][k] = acc_posteriors_means[j][k] / acc_posteriors_weights[j];
		}
	}

	/*
	// first the gaussians
	real* p_weights_acc = weights_acc;
	for (int i=0;i<n_gaussians;i++,p_weights_acc++) 
	{
		if (*p_weights_acc == 0) 
		{
			warning("Gaussian %d of GMM is not used in EM",i);
		}
		else
		{
			real* p_means_i = means[i];
			real* p_var_i = var[i];
			real* p_means_acc_i = means_acc[i];
			real* p_var_acc_i = var_acc[i];
			for (int j=0;j<n_inputs;j++) 
			{
				*p_means_i = *p_means_acc_i++ / *p_weights_acc;
				real v = *p_var_acc_i++ / *p_weights_acc - *p_means_i * *p_means_i++;
				*p_var_i++ = v >= var_threshold[j] ? v : var_threshold[j];
			}
		}
	}
	// then the weights
	real sum_weights_acc = 0;
	p_weights_acc = weights_acc;
	for (int i=0;i<n_gaussians;i++)
		sum_weights_acc += *p_weights_acc++;
	real *p_log_weights = log_weights;
	real log_sum = log(sum_weights_acc);
	p_weights_acc = weights_acc;
	for (int i=0;i<n_gaussians;i++)
		*p_log_weights++ = log(*p_weights_acc++) - log_sum;
	*/

	if(use_log)
	{
		// pre-compute constants
		double cst_ = n_inputs * THLog2Pi;
		for(int j = 0 ; j < n_means ; j++) 
		{
			double log_det = 0.0;
			for(int k = 0 ; k < n_inputs ; k++) log_det += log(variances[j][k]);
			g_norm[j] = cst_ + log_det;
		}
	}

	return true;
}
	
}

