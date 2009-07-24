#include "LRTrainer.h"
#include "LRMachine.h"

#ifdef HAVE_LBFGS
	#include "lbfgs.h"
#endif

static double getSign(double value)
{
	return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
}

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
	const int n_L1_priors = 7;
	const double L1_priors[n_L1_priors] = { 0.0001,
						0.0010,
						0.0100,
						0.1000,
						1.0000,
						10.000,
						100.00 };

	const int n_L2_priors = 7;
	const double L2_priors[n_L2_priors] = { 0.0001,
						0.0010,
						0.0100,
						0.1000,
						1.0000,
						10.000,
						100.00 };

//	// Optimize L1 and L2 prior values (in the same time) against the validation dataset
//	for (int i_l1 = 0; i_l1 < n_L1_priors; i_l1 ++)
//	{
//		for (int i_l2 = 0; i_l2 < n_L2_priors; i_l2 ++)
//		{
//			const double L1_prior = L1_priors[i_l1];
//			const double L2_prior = L2_priors[i_l2];
//
//			// Train
//			train(L1_prior, L2_prior, weights, gradients, fselected, size, verbose);
//			lr_machine->setThreshold(0.5);
//			lr_machine->setWeights(weights);
//
//			// Test if the new parameters are better
//			const double detection_rate = test(lr_machine, m_validation_dataset);
//			if (detection_rate > best_detection_rate)
//			{
//				for (int i = 0; i <= size; i ++)
//				{
//					best_weights[i] = weights[i];
//					best_fselected[i] = fselected[i];
//				}
//
//				best_detection_rate = detection_rate;
//				best_L1_prior = L1_prior;
//				best_L2_prior = L2_prior;
//			}
//
//			print("LRTrainer: L1_prior = %lf, L2_prior = %lf => detection rate = %lf.\n",
//				L1_prior, L2_prior, detection_rate);
//		}
//	}

	// Optimize L1 prior value against the validation dataset
	for (int i_l1 = 0; i_l1 < n_L1_priors; i_l1 ++)
	{
		const double L1_prior = L1_priors[i_l1];

		// Train
		train(L1_prior, best_L2_prior, weights, gradients, fselected, size, verbose);
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
		}

		print("LRTrainer: L1_prior = %lf, L2_prior = %lf => detection rate = %lf.\n",
			L1_prior, best_L2_prior, detection_rate);
	}

	// Optimize L2 prior value against the validation dataset
	for (int i_l2 = 0; i_l2 < n_L2_priors; i_l2 ++)
	{
		const double L2_prior = L2_priors[i_l2];

		// Train
		train(best_L1_prior, L2_prior, weights, gradients, fselected, size, verbose);
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
			best_L2_prior = L2_prior;
		}

		print("LRTrainer: L1_prior = %lf, L2_prior = %lf => detection rate = %lf.\n",
			best_L1_prior, L2_prior, detection_rate);
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
// Callback functions for the L-BFGS optimization library

#ifdef HAVE_LBFGS

struct LBFGS_Data
{
	LBFGS_Data(DataSet* dataset, bool* fselected, double l1_prior, double l2_prior)
		:	m_dataset(dataset),
			m_fselected(fselected),
			m_l1_prior(l1_prior),
			m_l2_prior(l2_prior)
	{
	}

	DataSet*	m_dataset;
	bool*		m_fselected;
	double		m_l1_prior;
	double		m_l2_prior;
};

static lbfgsfloatval_t evaluate(	void* instance,
					const lbfgsfloatval_t* x, lbfgsfloatval_t* g, const int n,
					const lbfgsfloatval_t step)
{
	LBFGS_Data* data = (LBFGS_Data*)instance;
	const int size = n - 1;

	lbfgsfloatval_t fx = 0.0;

	// Compute the loss function to minimize in the <x> point: - loglikelihood
	const long n_samples = data->m_dataset->getNoExamples();
	for (long s = 0; s < n_samples; s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)data->m_dataset->getExample(s);

		const double score = LRMachine::sigmoidEps((const double*)example->dataR(), x, size);
		const double label = ((const DoubleTensor*)data->m_dataset->getTarget(s))->get(0);

		fx -= label * log(score) + (1.0 - label) * log(1.0 - score);
	}
	fx *= n_samples == 0 ? 1.0 : 1.0 / ((double)n_samples);

	// Compute the loss function to minimize in the <x> point: regularization terms
	for (int i = 0; i <= size; i ++)
	{
		fx += data->m_l1_prior * getSign(x[i]) + data->m_l2_prior * x[i];
	}

	// Compute the gradient in the <x> point
	LRTrainer::getGradient(data->m_dataset, g, x, size, data->m_l1_prior, data->m_l2_prior);
	for (int i = 0; i <= size; i ++)
		if (data->m_fselected[i] == false)
		{
			g[i] = 0.0;
		}

	return fx;
}

//static int progress(void* instance, const lbfgsfloatval_t* x, const lbfgsfloatval_t* g, const lbfgsfloatval_t fx,
//		    const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step,
//		    int n, int k, int ls)
//{
//	printf("Iteration %d:\n", k);
////	printf("  fx = %f, x[0] = %f, x[1] = %f\n", fx, x[0], x[1]);
////	printf("  xnorm = %f, gnorm = %f, step = %f\n", xnorm, gnorm, step);
////	printf("\n");
//	return 0;
//}

#endif

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
		getGradient(m_dataset, gradients, weights, size, L1_prior, L2_prior);
		int i_max_gradient = 0;
		double max_gradient = 0.0;
		for (int i = 0; i <= size; i ++)
			if (fselected[i] == false)
			{
				const double abs_gradient = fabs(gradients[i]);
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

#ifdef HAVE_LBFGS		// L-BFGS optimization

		// Initialize the parameters for the L-BFGS optimization
		lbfgs_parameter_t param;
		lbfgs_parameter_init(&param);
		//param.orthantwise_c = 1;
		//param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;

		// Start the L-BFGS optimization; this will invoke the callback functions
		//	evaluate() and progress() when necessary
		lbfgsfloatval_t fx;
		LBFGS_Data data(m_dataset, fselected, L1_prior, L2_prior);
		lbfgs(size + 1, weights, &fx, evaluate, NULL, (void*)&data, &param);

#else				// Batch gradient descent

		const double eps = 0.00001;
		const int max_n_iters = 1000;
		double learning_rate = 0.5;

		// Batch gradient descent ...
		int n_iters = 0;
		double max_diff = 10000000000.0, last_max_diff = 10000000000.0;
		while ((n_iters ++) < max_n_iters && max_diff > eps && learning_rate > eps)
		{
			// Compute the gradient
			getGradient(m_dataset, gradients, weights, size, L1_prior, L2_prior);

			// Update the weights (Batch Gradient Descent)
			max_diff = 0.0;
			for (int i = 0; i <= size; i ++)
				if (fselected[i] == true)
				{
					const double diff = gradients[i] + 2.0 * L2_prior * weights[i];
					weights[i] -= learning_rate * diff;
					max_diff += fabs(diff);
				}

			// Decrease the learning rate if we're getting away from the solution
			if (max_diff > last_max_diff && n_iters > 1)
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
#endif
		///////////////////////////////////////////////////////////////////////
	}
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

void LRTrainer::getGradient(DataSet* dataset, double* gradients, const double* weights, int size, double L1_prior, double L2_prior)
{
	// Initialize
	for (int i = 0; i <= size; i ++)
	{
		gradients[i] = 0.0;
	}

	// Add the gradient of the negative loglikelihood
	for (long s = 0; s < dataset->getNoExamples(); s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)dataset->getExample(s);
		const double* data = (const double*)example->dataR();
		const double score = LRMachine::sigmoidEps(data, weights, size);

		const double label = ((const DoubleTensor*)dataset->getTarget(s))->get(0);

		const double factor = - (label - score);
		for (int i = 0; i < size; i ++)
		{
			gradients[i] += factor * data[i];
		}
		gradients[size] += factor * 1.0;
	}

	const double inv = dataset->getNoExamples() == 0 ? 1.0 : 1.0 / ((double)dataset->getNoExamples());
	for (int i = 0; i <= size; i ++)
	{
		gradients[i] *= inv;
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
