#include "LRTrainer.h"
#include "LRMachine.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

LRTrainer::LRTrainer()
	:	m_validation_dataset(0)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

LRTrainer::~LRTrainer()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Set the validation dataset

bool LRTrainer::setValidationData(DataSet* dataset)
{
	if (dataset == 0)
	{
		return false;
	}

	m_validation_dataset = dataset;
	return true;
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

	const int size = m_dataset->getExample(0)->size(0);

	// Resize the LR machine
	LRMachine* lr_machine = dynamic_cast<LRMachine*>(m_machine);
	if (lr_machine == 0)
	{
		print("LRTrainer::train - can only train LR machines!\n");
		return false;
	}
	if (lr_machine->resize(size) == false)
	{
		print("LRTrainer::train - could not resize LR machine!\n");
		return false;
	}

	// Prepare buffers (weights, batch gradients, feature selection flags)
	double*	weights = new double[size + 1];
	double* gradients = new double[size + 1];
	bool* fselected = new bool[size + 1];
	for (int i = 0; i <= size; i ++)
	{
		weights[i] = 0.0;
		fselected[i] = false;
	}

	// Set constants
	const double L1_prior = 0.1;	// regularization coefficients
	const double L2_prior = 0.0;	// for L1&L2 norms

	// Batch gradient descent with L1 and L2 regularization terms - Grafting method
	int n_fselected = 0;
	while (n_fselected < size)
	{
		// Get the feature with the maximum gradient
		getGradient(gradients, weights, size, L1_prior, L2_prior);
		int i_max_gradient = 0;
		double max_gradient = 0.0;
		for (int i = 0; i <= size; i ++)
			if (fselected[i] == false)
			{
				const double abs_gradient = abs(gradients[i]);
				if (abs_gradient > max_gradient)
				{
					max_gradient = abs_gradient;
					i_max_gradient = i;
				}
			}

		print("max_gradient = %lf for feature [%d/%d].\n", max_gradient, i_max_gradient + 1, size + 1);

		// Convergence test
		if (max_gradient < L1_prior)
		{
			break;
		}

		// Select the feature with the maximum gradient
		fselected[i_max_gradient] = true;
		n_fselected ++;

		// Optimizate the criterion using the selected features/weights
		const int max_n_iters = 100;
		double learning_rate = 0.5;
		const double eps = 0.00001;

		int n_iters = 0;
		double max_diff = 10000000000.0, last_max_diff = 10000000000.0;
		while ((n_iters ++) < max_n_iters && max_diff > eps && learning_rate > eps)
		{
			// Compute the gradient
			getGradient(gradients, weights, size, L1_prior, L2_prior);

			// Update the weights (Batch Gradient Descent)
			max_diff = 0.0;
			double max_weight = 0.0;
			for (int i = 0; i <= size; i ++)
				if (fselected[i] == true)
				{
					const double diff = gradients[i] + 2.0 * L2_prior * weights[i];
					weights[i] -= learning_rate * diff;
					max_diff = max(max_diff, fabs(diff));
					max_weight = max(max_weight, fabs(weights[i]));
				}

			// Decrease the learning rate if we're getting away from the solution
			if (max_diff > last_max_diff)
			{
				learning_rate *= 0.5;
			}
			last_max_diff = max_diff;

			print("[%d/%d] - max_diff = %lf, max_weight = %lf, learning_rate = %lf\n",
				n_iters, max_n_iters, max_diff, max_weight, learning_rate);
		}
	}

	print("LRTrainer: [%d/%d] features have been selected.\n", n_fselected, size + 1);
	for (int i = 0; i <= size; i ++)
	{
		print("LRTrainer: feature [%d/%d]: weight = %f, selected = %s\n",
			i + 1, size + 1, weights[i], fselected[i] == true ? "true" : "false");
	}

	// Set the parameters to the machine
	lr_machine->setThreshold(0.5);
	lr_machine->setWeights(weights);

	// Cleanup
	delete[] weights;
	delete[] gradients;
	delete[] fselected;

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the gradient for the Grafting method (negative loglikelihoods + regularization terms)

static double getSign(double value)
{
	return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

void LRTrainer::getGradient(double* gradients, const double* weights, int size, double L1_prior, double L2_prior)
{
	// Initialize
	for (int i = 0; i <= size; i ++)
	{
		gradients[i] = 0.0;
	}

	// Add the gradient of the negative loglikelihood
	for (long s = 0; s < m_dataset->getNoExamples(); s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)m_dataset->getExample(s);
		const double* data = (const double*)example->dataR();
		const double score = LRMachine::sigmoid(data, weights, size);

		const double label = ((const DoubleTensor*)m_dataset->getTarget(s))->get(0);

		const double factor = - (label - score);
		for (int i = 0; i < size; i ++)
		{
			gradients[i] += factor * data[i];
		}
		gradients[size] += factor * 1.0;
	}

	// Add the gradient of the L1 regularization term
	for (int i = 0; i <= size; i ++)
	{
		if (gradients[i] > L1_prior)
		{
			gradients[i] += L1_prior;
		}
		else if (gradients[i] < -L1_prior)
		{
			gradients[i] -= L1_prior;
		}
		else
		{
			gradients[i] += L1_prior * getSign(weights[i]);
		}
	}

	// Add the gradient of the L2 regularization term
	for (int i = 0; i <= size; i ++)
	{
		gradients[i] += 2.0 * L2_prior * weights[i];
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
