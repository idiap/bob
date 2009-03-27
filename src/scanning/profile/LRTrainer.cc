#include "LRTrainer.h"
#include "LRMachine.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

LRTrainer::LRTrainer()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

LRTrainer::~LRTrainer()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Train the given machine on the given dataset

bool LRTrainer::train()
{
	// Check parameters
	if (	m_machine == 0 ||
		m_dataset == 0 ||
		m_dataset->getNoExamples() < 1 ||
		m_dataset->getExample(0)->getDatatype() != Tensor::Double ||
		m_dataset->getExample(0)->nDimension() != 1 ||
		m_dataset->getTarget(0)->getDatatype() != Tensor::Double ||
		m_dataset->getTarget(0)->nDimension() != 1)
	{
		print("LRTrainer::train - invalid parameters!\n");
		return false;
	}

	LRMachine* lr_machine = dynamic_cast<LRMachine*>(m_machine);
	if (lr_machine == 0)
	{
		print("LRTrainer::train - can only train LR machines!\n");
		return false;
	}

	// Allocate the between&within class covariance and averages
	const int size = m_dataset->getExample(0)->size(0);
	if (lr_machine->resize(size) == false)
	{
		print("LRTrainer::train - could not resize LR machine!\n");
		return false;
	}

	double*	weights = new double[size + 1];
	double* buff = new double[size + 1];
	for (int i = 0; i <= size; i ++)
	{
		weights[i] = 0.0;
	}

	// Stochastic gradient descent with a prior on the weights (regularization term)
	const int max_n_iters = 10000;
	const double alpha = 0.5;	// learning rate
	const double beta = 0.1;	// prior over weights (regularization weight)
	const double eps = 0.00001;	// convergence criterion

	int n_iters = 0;
	double max_diff = 1000.0;
	while ((n_iters ++) < max_n_iters && max_diff > eps)
	{
		for (int i = 0; i <= size; i ++)
		{
			buff[i] = 0.0;
		}

		// Compute the gradient descent
		const int n_samples = m_dataset->getNoExamples();
		for (long s = 0; s < n_samples; s ++)
		{
			const DoubleTensor* example = (const DoubleTensor*)m_dataset->getExample(s);

			const double* data = (const double*)example->dataR();
			const double label = ((const DoubleTensor*)m_dataset->getExample(s))->get(0);

			const double score = LRMachine::sigmoid(data, weights, size);
			const double factor = alpha * (label - score);

			for (int i = 0; i < size; i ++)
			{
				buff[i] += factor * data[i];
			}
			buff[size] += factor * 1.0;
		}

		// Update the weights
		max_diff = 0.0;
		double max_weight = 0.0;
		for (int i = 0; i <= size; i ++)
		{
			const double diff = buff[i] - beta * weights[i];
			weights[i] += diff;
			max_diff = max(max_diff, diff);
			max_weight = max(max_weight, fabs(weights[i]));
		}

		//print("[%d/%d] - max_diff = %f, max_weight = %f\n", n_iters, max_n_iters, max_diff, max_weight);
	}

	// Set the parameters to the machine
	lr_machine->setThreshold(0.5);
	lr_machine->setWeights(weights);

//	print("weights:\n");
//	for (int i = 0; i <= size; i ++)
//	{
//		print("[%d] = %f\n", i, weights[i]);
//	}

	// Cleanup
	delete[] weights;
	delete[] buff;

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the loglikelihood for the given dataset

double LRTrainer::getLLH(const double* weights, int size) const
{
	double llh = 0.0;

	const int n_samples = m_dataset->getNoExamples();
	for (long s = 0; s < n_samples; s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)m_dataset->getExample(s);

		const double* data = (const double*)example->dataR();
		const double label = ((const DoubleTensor*)m_dataset->getExample(s))->get(0);

		if (label == 0.0)
		{
			llh += log(1.0 - LRMachine::sigmoid(data, weights, size));
		}
		else
		{
			llh += log(LRMachine::sigmoid(data, weights, size));
		}
	}

	return llh;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
