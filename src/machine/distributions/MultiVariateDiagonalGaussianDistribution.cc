#include "MultiVariateDiagonalGaussianDistribution.h"

namespace Torch {

MultiVariateDiagonalGaussianDistribution::MultiVariateDiagonalGaussianDistribution()
{
	addBOption("variance update", false, "update variances");
	addBOption("log mode", false, "use log for computing");

	use_log = false;
	
	//
	posterior_numerator = NULL;
	g_norm = NULL;
}

MultiVariateDiagonalGaussianDistribution::MultiVariateDiagonalGaussianDistribution(int n_inputs_, int n_gaussians_) : MultiVariateNormalDistribution(n_inputs_, n_gaussians_)
{
	addBOption("variance update", false, "update variances");
	addBOption("log computing", false, "use log for computing");
	
	use_log = false;

	//
	posterior_numerator = new double [n_gaussians];
	g_norm = new double [n_gaussians];
	for(int j = 0 ; j < n_gaussians ; j++) g_norm[j] = 0.0;
}

MultiVariateDiagonalGaussianDistribution::~MultiVariateDiagonalGaussianDistribution()
{
	if(posterior_numerator != NULL) delete []posterior_numerator;
	if(g_norm != NULL) delete []g_norm;
}

bool MultiVariateDiagonalGaussianDistribution::prepare()
{
	use_log = getBOption("log mode");

	if(use_log)
	{
		// pre-compute constants
		double cst_ = n_inputs * THLog2Pi;
		for(int j = 0 ; j < n_gaussians ; j++) 
		{
			double log_det = 0.0;
			for(int k = 0 ; k < n_inputs ; k++) log_det += log(variances[j][k]);
			g_norm[j] = cst_ + log_det;
		}
	}

	return ProbabilityDistribution::prepare();
}

bool MultiVariateDiagonalGaussianDistribution::EMinit()
{
	if(use_log)
	{
		// pre-compute constants
		double cst_ = n_inputs * THLog2Pi;
		for(int j = 0 ; j < n_gaussians ; j++) 
		{
			double log_det = 0.0;
			for(int k = 0 ; k < n_inputs ; k++) log_det += log(variances[j][k]);
			g_norm[j] = cst_ + log_det;
		}
	}

	acc_posteriors_sum_weights = 0.0;
	for(int j = 0 ; j < n_gaussians ; j++)
	{
		posterior_numerator[j] = 0.0;
		acc_posteriors_weights[j] = 0.0;

		acc_posteriors_means[j] = &buffer_acc_posteriors_means[j*n_inputs];
		for(int k = 0 ; k < n_inputs ; k++) acc_posteriors_means[j][k] = 0.0;

		acc_posteriors_variances[j] = &buffer_acc_posteriors_variances[j*n_inputs];
		for(int k = 0 ; k < n_inputs ; k++) acc_posteriors_variances[j][k] = 0.0;
	}

	return true;
}

void MultiVariateDiagonalGaussianDistribution::EMaccPosteriors(const DoubleTensor *input)
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
		for(int j = 0 ; j < n_gaussians ; j++) posterior_numerator[j] = log(weights[j]) + current_likelihood_one_gaussian[j];
	}
	else
	{
   		posterior_denominator = 0.0;

		for(int j = 0 ; j < n_gaussians ; j++) 
		{
			posterior_numerator[j] = weights[j] * sampleProbabilityOneGaussian(src, j); 
			posterior_denominator += posterior_numerator[j];
		}
	}

	// If you use measureExample below then you should use
	// this statement to compute the posterior instead of the
	// previous statement
   	/*
   	sampleProbability(data_[i]);

	double posterior_denominator = current_likelihood;
	for(int j = 0 ; j < n_gaussians ; j++) posterior_numerator[j] = weights[j] * current_likelihood_one_gaussian[j];
	*/

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
	for(int j = 0 ; j < n_gaussians ; j++)
	{
	   	double posterior_j;
	
		if(use_log) posterior_j = exp(posterior_numerator[j] - posterior_denominator);
		else posterior_j = posterior_numerator[j] / posterior_denominator;

		acc_posteriors_weights[j] += posterior_j;
		acc_posteriors_sum_weights += posterior_j;

		for(int k = 0 ; k < n_inputs ; k++) 
		{
			double z = src[k];

			acc_posteriors_means[j][k] += posterior_j * z;

			//double z = sample_[k] - means[j][k];
			acc_posteriors_variances[j][k] += posterior_j * z * z;
		}
	}
}

bool MultiVariateDiagonalGaussianDistribution::EMupdate()
{
	bool variance_update = getBOption("variance update");

	for(int j = 0 ; j < n_gaussians ; j++)
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

			if(variance_update)
			{
	   			/** Update rule (1) for variances:
					\begin{equation}
					\sigma_j = \frac{\sum_{i=1}^{n_samples} P(q_j | x_i) \times (x_i - \mu_j)}{\sum_{i=1}^{n_samples} P(q_j | x_i)}
					\end{equation}
				*/
				//double v = acc_posteriors_variances[j][k] / acc_posteriors_weights[j];
	   		
				/** Update rule (2) for variances:
					\begin{equation}
					\sigma_j = \frac{\sum_{i=1}^{n_samples} P(q_j | x_i) \times x_i^2}{\sum_{i=1}^{n_samples} P(q_j | x_i)} - \mu_j^2
					\end{equation}
				*/
				double v = acc_posteriors_variances[j][k] / acc_posteriors_weights[j] - means[j][k] * means[j][k];

				// variance flooring
				variances[j][k] = (v >= threshold_variances[k]) ? v : threshold_variances[k];
			}
		}
	}

	if(use_log)
	{
		// pre-compute constants
		double cst_ = n_inputs * THLog2Pi;
		for(int j = 0 ; j < n_gaussians ; j++) 
		{
			double log_det = 0.0;
			for(int k = 0 ; k < n_inputs ; k++) log_det += log(variances[j][k]);
			g_norm[j] = cst_ + log_det;
		}
	}

	return true;
}
	
double MultiVariateDiagonalGaussianDistribution::sampleProbability(double *sample_)
{
	if(use_log)
	{
		double max_ = -DBL_MAX;
		best_gaussian = -1;
		current_likelihood = THLogZero;

		for(int j = 0 ; j < n_gaussians ; j++)
		{
			sampleProbabilityOneGaussian(sample_, j);

		   	current_likelihood = THLogAdd(current_likelihood, log(weights[j]) + current_likelihood_one_gaussian[j]);
			
			if(current_likelihood_one_gaussian[j] > max_)
			{
				max_ = current_likelihood_one_gaussian[j];
				best_gaussian = j;
			}
		}
	}
	else
	{
		double max_ = -DBL_MAX;
		best_gaussian = -1;
		current_likelihood = 0.0;

		for(int j = 0 ; j < n_gaussians ; j++)
		{
			current_likelihood += weights[j] * sampleProbabilityOneGaussian(sample_, j);

			if(current_likelihood_one_gaussian[j] > max_)
			{
				max_ = current_likelihood_one_gaussian[j];
				best_gaussian = j;
			}
		}

		current_likelihood = log(current_likelihood);
	}

	return current_likelihood;
}

double MultiVariateDiagonalGaussianDistribution::sampleProbabilityOneGaussian(double *sample_, int g)
{
	double l;
	double z;
	double det;

	if(use_log)
	{
		z = 0.0;
		for(int k = 0 ; k < n_inputs ; k++)
		{
			double zz = sample_[k] - means[g][k];
			
			z += zz*zz / variances[g][k];
		}

		l = -0.5 * (g_norm[g] + z);
	}
	else
	{
		det = 1.0;
		z = 0.0;
		for(int k = 0 ; k < n_inputs ; k++)
		{
			double zz = sample_[k] - means[g][k];
			
			z += zz*zz / variances[g][k];
			det *= variances[g][k];
		}

		l = exp(-0.5*z) / sqrt(pow(2.0*M_PI, n_inputs) * det);
		// Warning !! the following formula is incorrect and produces a likelihood decrease during EM
		//l = exp(-0.5*z) / sqrt(pow(2.0*M_PI, dim)) * det;
	}

	current_likelihood_one_gaussian[g] = l;

	return l;
}

}

