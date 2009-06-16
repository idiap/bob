#include "MultiVariateNormalDistribution.h"

namespace Torch {

MultiVariateNormalDistribution::MultiVariateNormalDistribution()
{
	weights = NULL;
	means = NULL;
	variances = NULL;
	threshold_variances = NULL;
	current_likelihood_one_gaussian = NULL;
	acc_posteriors_weights = NULL;
	buffer_acc_posteriors_means = NULL;
	acc_posteriors_means = NULL;
	buffer_acc_posteriors_variances = NULL;
	acc_posteriors_variances = NULL;
}

MultiVariateNormalDistribution::MultiVariateNormalDistribution(int n_inputs_, int n_gaussians_) : ProbabilityDistribution(n_inputs_, n_gaussians_ * n_inputs_ * 2 + n_gaussians_)
{
   	//
	n_gaussians = n_gaussians_;

	weights = NULL;
	means = NULL;
	variances = NULL;
	threshold_variances = NULL;
	current_likelihood_one_gaussian = NULL;
	acc_posteriors_weights = NULL;
	buffer_acc_posteriors_means = NULL;
	acc_posteriors_means = NULL;
	buffer_acc_posteriors_variances = NULL;
	acc_posteriors_variances = NULL;

	//
	weights = parameters;

	//
	means = (double **) THAlloc(n_gaussians * sizeof(double *));
	variances = (double **) THAlloc(n_gaussians * sizeof(double *));
	double *p = &parameters[n_gaussians];
	for(int j = 0 ; j < n_gaussians ; j++)
	{
		means[j] = p; p += n_inputs;
		variances[j] = p; p += n_inputs;
	}
	
	//
	threshold_variances = (double *) THAlloc(n_inputs * sizeof(double));
	for(int i = 0 ; i < n_inputs ; i++) threshold_variances[i] = 0.0;
	
	//
	current_likelihood_one_gaussian = (double *) THAlloc(n_gaussians * sizeof(double));
	for(int j = 0 ; j < n_gaussians ; j++) current_likelihood_one_gaussian[j] = 0.0;

	//
	acc_posteriors_weights = (double *) THAlloc(n_gaussians * sizeof(double));
	buffer_acc_posteriors_means = (double *) THAlloc(n_gaussians * n_inputs * sizeof(double));
	acc_posteriors_means = (double **) THAlloc(n_gaussians * sizeof(double *));
	buffer_acc_posteriors_variances = (double *) THAlloc(n_gaussians * n_inputs * sizeof(double));
	acc_posteriors_variances = (double **) THAlloc(n_gaussians * sizeof(double *));
}

MultiVariateNormalDistribution::~MultiVariateNormalDistribution()
{
	THFree(acc_posteriors_variances);
	THFree(buffer_acc_posteriors_variances);
	THFree(acc_posteriors_means);
	THFree(buffer_acc_posteriors_means);
	THFree(acc_posteriors_weights);
	THFree(current_likelihood_one_gaussian);
	THFree(threshold_variances);
	THFree(variances);
	THFree(means);
}

bool MultiVariateNormalDistribution::forward(const DoubleTensor *input)
{
	double *src = (double *) input->dataR();
	double *dst = (double *) m_output.dataW();

	dst[0] = sampleProbability(src);

	return true;
}

/*
void MultiVariateNormalDistribution::initMeans(real **means_)
{
	// init only the means from provided means
	
	for(int j = 0 ; j < n_gaussians ; j++) 
	{
		for(int k = 0 ; k < dim ; k++) 
		{
			means[j][k] = means_[j][k];
			variances[j][k] = threshold_variances[k];
		}
		weights[j] = 1.0 / (real) n_gaussians;
	}
}

void MultiVariateNormalDistribution::initMeans(int n_data_, real **data_)
{
	// init only means from assigning a random sample per partitions

	if(n_gaussians > n_data_) warning("There is more gaussians than samples. This could creates some troubles.");

	int n_partitions = (int)(n_data_ / (real)n_gaussians);

	for(int j = 0 ; j < n_gaussians ; j++) 
	{
		int offset = j*n_partitions;
		int index = offset + (int)(Random::uniform()*(real)n_partitions);

		if(index < 0) warning("under limit");
		if(index >= n_data_) warning("over limit");

		for(int k = 0 ; k < dim ; k++) 
		{
			means[j][k] = data_[index][k];
			variances[j][k] = threshold_variances[k];
		}
		weights[j] = 1.0 / (real) n_gaussians;
	}

}

void MultiVariateNormalDistribution::initVariances(int n_data_, real **data_, real factor_variance_threshold_)
{
	// init only variances to the variance of data
	// Note: it could be interesting to compute the variance of samples for each cluster !

	real *mean = (real *) allocator->alloc(dim * sizeof(real));

	for(int k = 0 ; k < dim ; k++) mean[k] = threshold_variances[k] = 0.0;

	for(int i = 0 ; i < n_data_ ; i++)
	{
		for(int k = 0 ; k < dim ; k++)
		{
			real z = data_[i][k];

			mean[k] += z;
			threshold_variances[k] += z*z;
		}
	}

	for(int k = 0 ; k < dim ; k++)
	{
		mean[k] /= (real) n_data_;
		threshold_variances[k] /= (real) n_data_;
		threshold_variances[k] -= mean[k] * mean[k];
		if(threshold_variances[k] <= 0.0) threshold_variances[k] = 1.0;
		//else threshold_variances[k] = sqrt(threshold_variances[k]);
	}

	print("Mean of data: [ ");
	for(int k = 0 ; k < dim ; k++) print("%g ", mean[k]);
	print("]\n");

	print("Variance of data: [ ");
	for(int k = 0 ; k < dim ; k++) print("%g ", threshold_variances[k]);
	print("]\n");

	for(int j = 0 ; j < n_gaussians ; j++)
		for(int k = 0 ; k < dim ; k++)
			variances[j][k] = threshold_variances[k];

	for(int k = 0 ; k < dim ; k++)
		threshold_variances[k] *= factor_variance_threshold_;

	print("Variance threshold: [ ");
	for(int k = 0 ; k < dim ; k++) print("%g ", threshold_variances[k]);
	print("]\n");

	allocator->free(mean);
}
*/

bool MultiVariateNormalDistribution::shuffle()
{
   	m_parameters->print("MultiVariateNormalDistribution parameters");
	
   	double z = 0.0;

	Torch::print("n_gaussians = %d\n", n_gaussians);
	Torch::print("n_inputs = %d\n", n_inputs);
	      
	for(int j = 0 ; j < n_gaussians ; j++)
	{
	   	//Torch::print("w[%d] = %g\n", j, weights[j]);
		//weights[j] = THRandom_uniform(0, 1);
		//z += weights[j];

		for(int k = 0 ; k < n_inputs ; k++)
		{
			//means[j][k] = THRandom_uniform(0, 1);
			//variances[j][k] = THRandom_uniform(0, 1);
		}
	}

	//for(int j = 0 ; j < n_gaussians ; j++) weights[j] /= z;

   	m_parameters->print("MultiVariateNormalDistribution parameters");

	return true;
}

void MultiVariateNormalDistribution::print()
{
   	double z = 0.0;

	for(int j = 0 ; j < n_gaussians ; j++)
	{
		Torch::print("Gaussian [%d]\n", j);

		Torch::print("   weight = %g\n", weights[j]);
		z += weights[j];

		Torch::print("   mean = [ ");
		for(int k = 0 ; k < n_inputs ; k++) Torch::print("%g ", means[j][k]);
		Torch::print("]\n");

		Torch::print("   variance = [ ");
		for(int k = 0 ; k < n_inputs ; k++) Torch::print("%g ", variances[j][k]);
		Torch::print("]\n");
	}
	Torch::print("Sum weights = %g\n", z);
}
	
}

