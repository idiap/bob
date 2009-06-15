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
	const bool verbose = getBOption("verbose");

	// Check parameters
	if (	m_machine == 0 ||

		m_dataset == 0 ||
		m_dataset->getExampleType() != Tensor::Double ||
		m_dataset->getTargetType() != Tensor::Double ||
		m_dataset->getNoExamples() < 1 ||
		m_dataset->getExample(0)->nDimension() != 1 ||
		m_dataset->getTarget(0)->nDimension() != 1 ||

		m_validation_dataset == 0 ||
		m_validation_dataset->getExampleType() != m_dataset->getExampleType() ||
		m_validation_dataset->getTargetType() != m_dataset->getTargetType() ||
		m_validation_dataset->getNoExamples() < 1 ||
		m_validation_dataset->getExample(0)->nDimension() != m_dataset->getExample(0)->nDimension() ||
		m_validation_dataset->getTarget(0)->nDimension() != m_dataset->getTarget(0)->nDimension())
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

//	// TEST!!!
//	for (long s = 0; s < m_dataset->getNoExamples(); s ++)
//	{
//		print("[%d/%d]: ", s, m_dataset->getNoExamples());
//		const DoubleTensor* example = (const DoubleTensor*)m_dataset->getExample(s);
//
//		for (int i = 0; i < size; i ++)
//		{
//			print("%lf\t", example->get(i));
//		}
//
//		print("-> %lf\n", ((DoubleTensor*)m_dataset->getTarget(s))->get(0));
//	}

	// Prepare buffers (weights, batch gradients, feature selection flags)
	double*	weights = new double[size + 1];
	double* gradients = new double[size + 1];
	bool* fselected = new bool[size + 1];

	double*	best_weights = new double[size + 1];
	bool* best_fselected = new bool[size + 1];
	double best_detection_rate = 0.0;
	double best_L1_prior = 0.0;
	double best_L2_prior = 0.0;

	// Set constants
	const int n_L1_priors = 22;
	const double L1_priors[n_L1_priors] = { 0.0,
						0.001, 0.002, 0.005,
						0.01, 0.02, 0.05,
						0.1, 0.2, 0.5,
						1.0, 2.0, 5.0,
						10.0, 20.0, 50.0,
						100.0, 200.0, 500.0,
						1000.0, 2000.0, 5000.0 };

	const int n_L2_priors = 22;
	const double L2_priors[n_L2_priors] = { 0.0,
						0.001, 0.002, 0.005,
						0.01, 0.02, 0.05,
						0.1, 0.2, 0.5,
						1.0, 2.0, 5.0,
						10.0, 20.0, 50.0,
						100.0, 200.0, 500.0,
						1000.0, 2000.0, 5000.0 };

	// Vary the L1 and L2 priors and choose the best values testing against the validation dataset
	for (int i_l1 = 0; i_l1 < n_L1_priors; i_l1 ++)
		for (int i_l2 = 0; i_l2 < n_L2_priors; i_l2 ++)
		{
			const double L1_prior = L1_priors[i_l1];
			const double L2_prior = L2_priors[i_l2];

			// Train
			train(L1_prior, L2_prior, weights, gradients, fselected, size, verbose);
			lr_machine->setThreshold(0.5);
			lr_machine->setWeights(weights);

			// Test if the new parameters are better
			const double detection_rate = test(lr_machine, m_validation_dataset);
			if (detection_rate > best_detection_rate)
			{
				for (int i = 0; i <= size; i ++)
				{
					best_weights[i] = weights[i];
					best_fselected[i] = fselected[i];
				}

				best_detection_rate = detection_rate;
				best_L1_prior = L1_prior;
				best_L2_prior = L2_prior;
			}

			print("LRTrainer: L1_prior = %lf, L2_prior = %lf => detection rate = %lf.\n",
				L1_prior, L2_prior, detection_rate);
		}

	// Debug
	if (verbose == true)
	{
		int n_fselected = 0;
		for (int i = 0; i <= size; i ++)
		{
			if (best_fselected[i] == true)
			{
				n_fselected ++;
			}
		}

		print("LRTrainer: -----------------------------------------------------------\n");
		print("LRTrainer: optimum L1_prior = %lf, L2_prior = %lf.\n", best_L1_prior, best_L2_prior);
		print("LRTrainer: [%d/%d] features have been selected.\n", n_fselected, size + 1);
		for (int i = 0; i <= size; i ++)
		{
			print("LRTrainer: feature [%d/%d]: weight = %f, selected = %s\n",
				i + 1, size + 1, best_weights[i], best_fselected[i] == true ? "true" : "false");
		}
		print("LRTrainer: => detection rate on validation dataset = %lf%%.\n", best_detection_rate);
		print("LRTrainer: -----------------------------------------------------------\n");
	}

	// Set the parameters to the machine
	lr_machine->setThreshold(0.5);
	lr_machine->setWeights(best_weights);

	// Cleanup
	delete[] weights;
	delete[] gradients;
	delete[] fselected;
	delete[] best_weights;
	delete[] best_fselected;

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Train the LR machine using the given L1 and L2 priors

void LRTrainer::train(	double L1_prior, double L2_prior,
			double* weights, double* gradients, bool* fselected, int size, bool verbose)
{
	// Prepare buffers (weights, batch gradients, feature selection flags)
	for (int i = 0; i <= size; i ++)
	{
		weights[i] = 0.0;
		fselected[i] = false;
	}

	// Batch gradient descent with L1 and L2 regularization terms - Grafting method
	int n_fselected = 0;
	while (n_fselected <= size)
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

		// Convergence test
		if (max_gradient < L1_prior)
		{
			break;
		}

		// Select the feature with the maximum gradient
		fselected[i_max_gradient] = true;
		n_fselected ++;

//		// Debug
//		if (verbose == true)
//		{
//			print("\tLRTrainer: the %dth selected feature is [%d]. Optimizing ...\n",
//				n_fselected, i_max_gradient);
//		}

		///////////////////////////////////////////////////////////////////////
		// Optimize the criterion using the selected features/weights

		const int max_n_iters = 1000;
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
			for (int i = 0; i <= size; i ++)
				if (fselected[i] == true)
				{
					const double diff = gradients[i] + 2.0 * L2_prior * weights[i];
					weights[i] -= learning_rate * diff;
					max_diff = max(max_diff, fabs(diff));
				}

			// Decrease the learning rate if we're getting away from the solution
			if (max_diff > last_max_diff)
			{
				learning_rate *= 0.5;
			}
			last_max_diff = max_diff;

//			// Debug
//			if (verbose == true)
//			{
//				print("\tLRTrainer: [%d/%d] iters - max_diff = %lf, learning_rate = %lf\n",
//					n_iters, max_n_iters, max_diff, learning_rate);
//			}
		}
		///////////////////////////////////////////////////////////////////////
	}
//
//	// Debug
//	if (verbose == true)
//	{
//		print("\tLRTrainer: -----------------------------------------------------------\n");
//		print("\tLRTrainer: [%d/%d] features have been selected.\n", n_fselected, size + 1);
//		for (int i = 0; i <= size; i ++)
//		{
//			print("\tLRTrainer: feature [%d/%d]: weight = %f, selected = %s\n",
//				i + 1, size + 1, weights[i], fselected[i] == true ? "true" : "false");
//		}
//		print("\tLRTrainer: -----------------------------------------------------------\n");
//	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Test the LR machine (returns the detection rate in percentages)

double LRTrainer::test(LRMachine* machine, DataSet* samples)
{
	const double* machine_output = (const double*)(machine->getOutput().dataR());
	const double threshold = machine->getThreshold();

	long correct = 0;
	for (long s = 0; s < samples->getNoExamples(); s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)samples->getExample(s);
		CHECK_FATAL(machine->forward(*example) == true);

		if (	(((DoubleTensor*)samples->getTarget(s))->get(0) >= threshold) ==
			(*machine_output >= threshold))
		{
			correct ++;
		}
	}

	return 100.0 * (correct + 0.0) / (samples->getNoExamples() == 0 ? 1.0 : samples->getNoExamples());
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
