#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Build a random sample/target
///////////////////////////////////////////////////////////////////////////

void buildSample(DoubleTensor* example, int n_dims)
{
	example->resize(n_dims);
	for (int i = 0; i < n_dims; i ++)
	{
		example->set(i, (double)(rand() % 512 - 256.0));
	}
}

void buildTarget(const DoubleTensor* example, DoubleTensor* target,
		int n_dims, double* weights, double bias, int noise_mean, int noise_variance)
{
	double sum = 0.0;
	for (int i = 0; i < n_dims; i ++)
	{
		if (i % 2 == 0)
		{
			sum += example->get(i) * weights[i];
		}
	}
	sum += bias + noise_mean + (rand() % noise_variance) * (rand() % 2 == 0 ? -1 : 1);

	target->resize(1);
	target->set(0, sum > 0.0 ? 0.1 : 0.9);
}

///////////////////////////////////////////////////////////////////////////
// Build a dataset of samples
///////////////////////////////////////////////////////////////////////////

MemoryDataSet* buildSamples(long n_samples, int n_dims, double* weights, double bias, int noise_mean, int noise_variance)
{
	MemoryDataSet* samples = manage(new MemoryDataSet(n_samples, Tensor::Double, true, Tensor::Double));

	DoubleTensor* targets = manage_array(new DoubleTensor[n_samples]);
	for (long s = 0; s < n_samples; s ++)
	{
		DoubleTensor* example = (DoubleTensor*)samples->getExample(s);
		buildSample(example, n_dims);

		buildTarget(example, &targets[s], n_dims, weights, bias, noise_mean, noise_variance);
		samples->setTarget(s, &targets[s]);
	}

	return samples;
}

///////////////////////////////////////////////////////////////////////////
// Count how many positive and negative samples are in a dataset
///////////////////////////////////////////////////////////////////////////

int countPositive(MemoryDataSet* samples)
{
	int cnt = 0;
	for (long s = 0; s < samples->getNoExamples(); s ++)
	{
		if (((DoubleTensor*)samples->getTarget(s))->get(0) > 0.5)
		{
			cnt ++;
		}
	}

	return cnt;
}

int countNegative(MemoryDataSet* samples)
{
	return samples->getNoExamples() - countPositive(samples);
}

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	//srand((unsigned int)time(0));
	srand(0);

	// Use a linear model (y = weights * random(x) + gaussian noise) to generate samples
	const long n_train_samples = 3000;//100 + rand() % 1024;
	const long n_valid_samples = 3000;//100 + rand() % 1024;
	const long n_test_samples = 3000;//100 + rand() % 1024;
	const int n_dims = 32;//3 + rand() % 32;

	const int noise_mean = 0;
	const int noise_variance = 1 + rand() % 5;

	const double bias = rand() % 512 - 256.0;
	double* weights = manage_array(new double[n_dims]);
	for (int i = 0; i < n_dims; i ++)
	{
		weights[i] = rand() % 32 - 16.0;
	}

	// Build the training, validation and testing datasets
	MemoryDataSet* train_samples = buildSamples(n_train_samples, n_dims, weights, bias, noise_mean, noise_variance);
	MemoryDataSet* valid_samples = buildSamples(n_valid_samples, n_dims, weights, bias, noise_mean, noise_variance);
	MemoryDataSet* test_samples = buildSamples(n_test_samples, n_dims, weights, bias, noise_mean, noise_variance);

	print("Created samples of [%d] dimensions:\n", n_dims);
	print("\t[%d] training samples: [%d] positive and [%d] negative.\n",
		n_train_samples, countPositive(train_samples), countNegative(train_samples));
	print("\t[%d] validation samples: [%d] positive and [%d] negative.\n",
		n_valid_samples, countPositive(valid_samples), countNegative(valid_samples));
	print("\t[%d] test samples: [%d] positive and [%d] negative.\n",
		n_test_samples, countPositive(test_samples), countNegative(test_samples));
//	print("\tLinear model with bias [%lf].\n", bias);
//	print("\tGaussian noise with mean [%d] and stdev. [%d].\n", noise_mean, noise_variance);
	print("--------------------------------------------------------------------\n");

	// Train LR
	print("Training Logistic Regression (LR) on [%d] samples of [%d] dimensions ...\n",
		n_train_samples, n_dims);

	LRMachine machine;
	LRTrainer trainer;

	CHECK_FATAL(trainer.setMachine(&machine) == true);
	CHECK_FATAL(trainer.setData(train_samples) == true);
	CHECK_FATAL(trainer.setValidationData(valid_samples) == true);
	CHECK_FATAL(trainer.setBOption("verbose", true) == true);
	CHECK_FATAL(trainer.setBOption("useL1", false) == true);
	CHECK_FATAL(trainer.setDOption("FARvsFRRRatio", 1.0) == true);
	CHECK_FATAL(trainer.train() == true);

	print("--------------------------------------------------------------------\n");

	double TAR, FAR, HTER;

	// Test LR
	print("Testing Logistic Regression (LR)\n");
	LRTrainer::test(&machine, train_samples, TAR, FAR, HTER, 1.0);
	print("\t[%d] training samples: TAR = %lf, FAR = %lf, HTER = %lf\n", n_train_samples, TAR, FAR, HTER);
	LRTrainer::test(&machine, valid_samples, TAR, FAR, HTER, 1.0);
	print("\t[%d] validation samples: TAR = %lf, FAR = %lf, HTER = %lf\n", n_valid_samples, TAR, FAR, HTER);
	LRTrainer::test(&machine, test_samples, TAR, FAR, HTER, 1.0);
	print("\t[%d] testing samples: TAR = %lf, FAR = %lf, HTER = %lf\n", n_test_samples, TAR, FAR, HTER);
	print("--------------------------------------------------------------------\n");

	return 0;
}
