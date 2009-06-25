#include "MultiVariateMeansDistribution.h"

namespace Torch {

MultiVariateMeansDistribution::MultiVariateMeansDistribution()
{
}

MultiVariateMeansDistribution::MultiVariateMeansDistribution(int n_inputs_, int n_means_) : MultiVariateNormalDistribution(n_inputs_, n_means_)
{
}

MultiVariateMeansDistribution::~MultiVariateMeansDistribution()
{
}

bool MultiVariateMeansDistribution::EMinit()
{
	float min_weights = getFOption("min weights");

	acc_posteriors_sum_weights = 0.0;
	for(int j = 0 ; j < n_means ; j++)
	{
		acc_posteriors_weights[j] = 0.0;

		for(int k = 0 ; k < n_inputs ; k++) acc_posteriors_means[j][k] = min_weights;
	}

	return true;
}

bool MultiVariateMeansDistribution::EMaccPosteriors(const DoubleTensor *input, const double input_posterior)
{
	double *src = (double *) input->dataR();

	sampleProbability(src);

	acc_posteriors_weights[best_mean]++;
	acc_posteriors_sum_weights ++;

	for(int k = 0 ; k < n_inputs ; k++) 
	{
		double z = src[k];

		acc_posteriors_means[best_mean][k] += z;
	}

	return true;
}

bool MultiVariateMeansDistribution::EMupdate()
{
	for(int j = 0 ; j < n_means ; j++)
	{
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

	return true;
}
	
double MultiVariateMeansDistribution::sampleProbability(double *sample_)
{
	double min_ = DBL_MAX;
	best_mean = -1;
	current_likelihood = 0.0;

	for(int j = 0 ; j < n_means ; j++)
	{
		double d = 0.0;
		for(int k = 0 ; k < n_inputs ; k++)
		{
			double z = sample_[k] - means[j][k];
			d += z*z;
		}
		
		current_likelihood_one_mean[j] = -d;
		
		if(d < min_)
		{
			min_ = d;
			best_mean = j;
		}
	}

	current_likelihood = -min_;

	return current_likelihood;
}

double MultiVariateMeansDistribution::sampleProbabilityOneMean(double *sample_, int m)
{
	double d = 0.0;
	for(int k = 0 ; k < n_inputs ; k++)
	{
		double z = sample_[k] - means[m][k];
		d += z*z;
	}
		
	current_likelihood_one_mean[m] = -d;

	return -d;
}

bool MultiVariateMeansDistribution::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("MultiVariateMeansDistribution::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("MultiVariateMeansDistribution::load - invalid <ID>, this is not the appropriate model!\n");
		return false;
	}

	if(m_parameters->loadFile(file) == false)
	{
	        Torch::message("MultiVariateMeansDistribution::load - failed to load parameters\n");
		return false;
	}

	n_inputs = m_parameters->getI("n_inputs");
	n_means = m_parameters->getI("n_means");

	MultiVariateNormalDistribution::cleanup();
   	cleanup();
	MultiVariateNormalDistribution::resize(n_inputs, n_means);
	resize(n_inputs, n_means);

	return true;
}

bool MultiVariateMeansDistribution::saveFile(File& file) const
{
	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("MultiVariateMeansDistribution::save - failed to write <ID> field!\n");
		return false;
	}

	if(m_parameters->saveFile(file) == false)
	{
	        Torch::message("MultiVariateMeansDistribution::load - failed to write parameters\n");
		return false;
	}

	return true;
}

}

