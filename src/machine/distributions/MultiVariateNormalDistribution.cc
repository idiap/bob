#include "MultiVariateNormalDistribution.h"

namespace Torch {

MultiVariateNormalDistribution::MultiVariateNormalDistribution()
{
	//
	n_means = 0;
	means = NULL;
	weights = NULL;

	//
	acc_posteriors_weights = NULL;
	buffer_acc_posteriors_means = NULL;
	acc_posteriors_means = NULL;
	current_likelihood_one_mean = NULL;

	//
	m_parameters->addI("n_inputs", 0, "number of dimensions of the multi-variate normal distribution");
	m_parameters->addI("n_means", 0, "number of means of the multi-variate normal distribution");
	m_parameters->addDarray("weigths", 0, 0.0, "weights of the multi-variate normal distribution");
	m_parameters->addDarray("means", 0, 0.0, "means of the multi-variate normal distribution");
}

MultiVariateNormalDistribution::MultiVariateNormalDistribution(int n_inputs_, int n_means_) : ProbabilityDistribution(n_inputs_)
{
   	//
	n_means = n_means_;
	means = NULL;
	weights = NULL;

	acc_posteriors_weights = NULL;
	buffer_acc_posteriors_means = NULL;
	acc_posteriors_means = NULL;
	current_likelihood_one_mean = NULL;

	//
	m_parameters->addI("n_inputs", n_inputs, "number of dimensions of the multi-variate normal distribution");
	m_parameters->addI("n_means", n_means, "number of means of the multi-variate normal distribution");
	m_parameters->addDarray("weigths", n_means, 0.0, "weights of the multi-variate normal distribution");
	m_parameters->addDarray("means", n_means*n_inputs, 0.0, "means of the multi-variate normal distribution");

	//
	resize(n_inputs_, n_means_);
}

bool MultiVariateNormalDistribution::resize(int n_inputs_, int n_means_)
{
	//Torch::print("MultiVariateNormalDistribution::resize(%d, %d)\n", n_inputs_, n_means_);
	
	//
	weights = m_parameters->getDarray("weigths");
	double *means_ = m_parameters->getDarray("means");
	means = (double **) THAlloc(n_means_ * sizeof(double *));
	double *p = means_;
	for(int j = 0 ; j < n_means_ ; j++)
	{
		means[j] = p; 
		p += n_inputs_;
	}

	//
	current_likelihood_one_mean = (double *) THAlloc(n_means_ * sizeof(double));
	for(int j = 0 ; j < n_means_ ; j++) current_likelihood_one_mean[j] = 0.0;

	//
	acc_posteriors_weights = (double *) THAlloc(n_means_ * sizeof(double));
	buffer_acc_posteriors_means = (double *) THAlloc(n_means_ * n_inputs_ * sizeof(double));
	acc_posteriors_means = (double **) THAlloc(n_means_ * sizeof(double *));

	for(int j = 0 ; j < n_means_ ; j++)
		acc_posteriors_means[j] = &buffer_acc_posteriors_means[j*n_inputs_];

	return true;
}

bool MultiVariateNormalDistribution::cleanup()
{
	//Torch::print("MultiVariateNormalDistribution::cleanup()\n");

	if(acc_posteriors_means != NULL) THFree(acc_posteriors_means);
	if(buffer_acc_posteriors_means != NULL) THFree(buffer_acc_posteriors_means);
	if(acc_posteriors_weights != NULL) THFree(acc_posteriors_weights);
	if(current_likelihood_one_mean != NULL) THFree(current_likelihood_one_mean);
	if(means != NULL) THFree(means);

	return true;
}

MultiVariateNormalDistribution::~MultiVariateNormalDistribution()
{	
	cleanup();
}

bool MultiVariateNormalDistribution::forward(const DoubleTensor *input)
{
	double *src = (double *) input->dataR();
	double *dst = (double *) m_output.dataW();

	dst[0] = sampleProbability(src);

	return true;
}

bool MultiVariateNormalDistribution::setMeans(double **means_)
{
	for(int j = 0 ; j < n_means ; j++) 
	{
		for(int k = 0 ; k < n_inputs ; k++) means[j][k] = means_[j][k];
		weights[j] = 1.0 / (double) n_means;
	}

	return true;
}

bool MultiVariateNormalDistribution::setMeans(DataSet *dataset_)
{
	// init only means from assigning a random sample per partitions
	
	if(dataset_ == NULL) return false;

	int n_data = dataset_->getNoExamples();
	if(n_means > n_data) warning("MultiVariateNormalDistribution::setMeans() There are more means than samples. This could creates some troubles.");

	int n_partitions = (int)(n_data / (double) n_means);

	for(int j = 0 ; j < n_means ; j++) 
	{
		int offset = j*n_partitions;
		int index = offset + (int)(THRandom_uniform(0, 1)*(double) n_partitions);

		if(index < 0) warning("under limit");
		if(index >= n_data) warning("over limit");

		Tensor *example = dataset_->getExample(index);
		if (	example->nDimension() != 1 || example->getDatatype() != Tensor::Double)
		{
			warning("MultiVariateNormalDistribution::setMeans() : incorrect number of dimensions or type.");
			break;
		}
		if (	example->size(0) != n_inputs)
		{
			warning("MultiVariateNormalDistribution::setMeans() : incorrect input size along dimension 0 (%d != %d).", example->size(0), n_inputs);
			break;
		}

		DoubleTensor *t_input = (DoubleTensor *) example;
		double *src = (double *) t_input->dataR();

		for(int k = 0 ; k < n_inputs ; k++) means[j][k] = src[k];
		weights[j] = 1.0 / (double) n_means;
	}

	return true;
}


bool MultiVariateNormalDistribution::shuffle()
{
	//Torch::print("MultiVariateNormalDistribution::shuffle()\n");
	
   	double z = 0.0;

	for(int j = 0 ; j < n_means ; j++)
	{
		weights[j] = THRandom_uniform(0, 1);
		z += weights[j];

		for(int k = 0 ; k < n_inputs ; k++)
			means[j][k] = THRandom_uniform(0, 1);
	}

	for(int j = 0 ; j < n_means ; j++) weights[j] /= z;

	return true;
}

bool MultiVariateNormalDistribution::print()
{
   	double z = 0.0;

	for(int j = 0 ; j < n_means ; j++)
	{
		Torch::print("Mean [%d]\n", j);

		Torch::print("   weight = %g\n", weights[j]);
		z += weights[j];

		Torch::print("   mean = [ ");
		for(int k = 0 ; k < n_inputs ; k++) Torch::print("%g ", means[j][k]);
		Torch::print("]\n");
	}
	Torch::print("Sum weights = %g\n", z);

	return true;
}
	
}

