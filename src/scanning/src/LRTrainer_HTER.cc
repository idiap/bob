#include "scanning/LRTrainer.h"
#include "scanning/LRMachine.h"
#include "measurer/measurer.h"

#ifdef HAVE_LBFGS
	#include "lbfgs/lbfgs.h"
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
	double* buf_gradients = new double[size + 1];
	bool* fselected = new bool[size + 1];

	double*	best_weights = new double[size + 1];
	bool* best_fselected = new bool[size + 1];
	double best_HTER = 1.0;
	double best_L1_prior = 0.0;
	double best_L2_prior = 0.0;
	double best_threshold = 0.5;

	// Set constants
	const int n_L1_priors = 19;
	const double L1_priors[n_L1_priors] = { 0.0001,
						0.0002,
						0.0005,
						0.0010,
						0.0020,
						0.0050,
						0.0100,
						0.0200,
						0.0500,
						0.1000,
						0.2000,
						0.5000,
						1.0000,
						2.0000,
						5.0000,
						10.000,
						20.000,
						50.000,
						100.00 };


	const int n_L2_priors = 19;
	const double L2_priors[n_L2_priors] = { 0.0001,
						0.0002,
						0.0005,
						0.0010,
						0.0020,
						0.0050,
						0.0100,
						0.0200,
						0.0500,
						0.1000,
						0.2000,
						0.5000,
						1.0000,
						2.0000,
						5.0000,
						10.000,
						20.000,
						50.000,
						100.00 };

	// Optimize L1 prior value against the validation dataset
	for (int i_l1 = 0; i_l1 < n_L1_priors; i_l1 ++)
	{
		const double L1_prior = L1_priors[i_l1];

		// Train
		if (train(L1_prior, best_L2_prior, weights, gradients, buf_gradients, fselected, size, verbose) == false)
		{
			continue;
		}
		lr_machine->setWeights(weights);
		optimize(lr_machine, m_validation_dataset);

		// Test if the new parameters are better
		double TAR, FAR, HTER;
		test(lr_machine, m_validation_dataset, TAR, FAR, HTER);
		if (HTER < best_HTER)
		{
			for (int i = 0; i <= size; i ++)
			{
				best_weights[i] = weights[i];
				best_fselected[i] = fselected[i];
			}

			best_HTER = HTER;
			best_L1_prior = L1_prior;
			best_threshold = lr_machine->getThreshold();
		}

		if (verbose == true)
		{
			print("LRTrainer: L1_prior = %lf, L2_prior = %lf => TAR = %lf, FAR = %lf, HTER = %lf, threshold = %lf.\n",
				L1_prior, best_L2_prior, TAR, FAR, HTER, lr_machine->getThreshold());
		}
	}

	// Optimize L2 prior value against the validation dataset
	for (int i_l2 = 0; i_l2 < n_L2_priors; i_l2 ++)
	{
		const double L2_prior = L2_priors[i_l2];

		// Train
		if (train(best_L1_prior, L2_prior, weights, gradients, buf_gradients, fselected, size, verbose) == false)
		{
			continue;
		}
		lr_machine->setWeights(weights);
		optimize(lr_machine, m_validation_dataset);

		// Test if the new parameters are better
		double TAR, FAR, HTER;
		test(lr_machine, m_validation_dataset, TAR, FAR, HTER);
		if (HTER < best_HTER)
		{
			for (int i = 0; i <= size; i ++)
			{
				best_weights[i] = weights[i];
				best_fselected[i] = fselected[i];
			}

			best_HTER = HTER;
			best_L2_prior = L2_prior;
			best_threshold = lr_machine->getThreshold();
		}

		if (verbose == true)
		{
			print("LRTrainer: L1_prior = %lf, L2_prior = %lf => TAR = %lf, FAR = %lf, HTER = %lf, threshold = %lf.\n",
				best_L1_prior, L2_prior, TAR, FAR, HTER, lr_machine->getThreshold());
		}
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
		print("LRTrainer: => HTER on validation dataset = %lf.\n", best_HTER);
		print("LRTrainer: -----------------------------------------------------------\n");
	}

	// Set the parameters to the machine
	lr_machine->setThreshold(best_threshold);
	lr_machine->setWeights(best_weights);

	// Cleanup
	delete[] weights;
	delete[] gradients;
	delete[] buf_gradients;
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
	LBFGS_Data(DataSet* dataset, double* buf_gradients, bool* fselected, double l1_prior, double l2_prior)
		:	m_dataset(dataset),
			m_buf_gradients(buf_gradients),
			m_fselected(fselected),
			m_l1_prior(l1_prior),
			m_l2_prior(l2_prior)
	{
	}

	DataSet*	m_dataset;
	double* 	m_buf_gradients;
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

	lbfgsfloatval_t fx = 0.0;		// Accumulate from positive samples
	lbfgsfloatval_t fx_neg = 0.0;		// Accumulate from negative samples

	// Compute the loss function to minimize in the <x> point: - loglikelihood
	//	(normalize positive and negative samples separetely)
	DataSet* dataset = data->m_dataset;
	for (long s = 0; s < dataset->getNoExamples(); s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)dataset->getExample(s);

		const double score = LRMachine::sigmoidEps((const double*)example->dataR(), x, size);
		const double label = ((const DoubleTensor*)dataset->getTarget(s))->get(0);
		double* dst = label > 0.5 ? &fx : &fx_neg;

		*dst -= label * log(score) + (1.0 - label) * log(1.0 - score);
	}

	double inv_n_pos, inv_n_neg;
	LRTrainer::getInvPosNeg(dataset, inv_n_pos, inv_n_neg);

	fx = fx * inv_n_pos + fx_neg * inv_n_neg;

	// Compute the loss function to minimize in the <x> point: regularization terms
	for (int i = 0; i <= size; i ++)
	{
		fx += data->m_l1_prior * fabs(x[i]) + data->m_l2_prior * x[i] * x[i];
	}

	// Compute the gradient in the <x> point
	LRTrainer::getGradient(data->m_dataset, g, data->m_buf_gradients, x, size, data->m_l1_prior, data->m_l2_prior);
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

bool LRTrainer::train(	double L1_prior, double L2_prior,
			double* weights,
			double* gradients, double* buf_gradients,
			bool* fselected, int size, bool verbose)
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
		getGradient(m_dataset, gradients, buf_gradients, weights, size, L1_prior, L2_prior);
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
		LBFGS_Data data(m_dataset, buf_gradients, fselected, L1_prior, L2_prior);
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
			getGradient(m_dataset, gradients, buf_gradients, weights, size, L1_prior, L2_prior);

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

	return n_fselected > 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Test the LR machine

void LRTrainer::test(LRMachine* machine, DataSet* samples, double& TAR, double& FAR, double& HTER)
{
	const double* machine_output = (const double*)(machine->getOutput().dataR());
	const double threshold = machine->getThreshold();

	long passed_pos = 0, passed_neg = 0;
	for (long s = 0; s < samples->getNoExamples(); s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)samples->getExample(s);
		CHECK_FATAL(machine->forward(*example) == true);

		if (*machine_output >= threshold)
		{
			const double label = ((const DoubleTensor*)samples->getTarget(s))->get(0);
			long* dst = label >= threshold ? &passed_pos : &passed_neg;
			(*dst) ++;
		}
	}

	double inv_n_pos, inv_n_neg;
	getInvPosNeg(samples, inv_n_pos, inv_n_neg);

	TAR = (double)passed_pos * inv_n_pos;
	FAR = (double)passed_neg * inv_n_neg;
	HTER = 0.5 * (FAR + 1.0 - TAR);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Optimize (minimum HTER) and test the LR machine
// NB: the optimum threshold is set to the machine!

void LRTrainer::optimize(LRMachine* machine, DataSet* samples)
{
//	{
//		double crt_TAR, crt_FAR, crt_HTER;
//		machine->setThreshold(0.5);
//		test(machine, samples, crt_TAR, crt_FAR, crt_HTER);
//		print("\tBEFORE: TAR = %lf, FAR = %lf, HTER = %lf\n", crt_TAR, crt_FAR, crt_HTER);
//	}

	const long n_samples = samples->getNoExamples();
	const double* machine_output = (const double*)(machine->getOutput().dataR());

	// Compute the scores and sort them
	LabelledMeasure* scores = new LabelledMeasure[n_samples];
	for (long s = 0; s < n_samples; s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)samples->getExample(s);
		const double label = ((const DoubleTensor*)samples->getTarget(s))->get(0);

		CHECK_FATAL(machine->forward(*example) == true);
		scores[s].measure = *machine_output;
		scores[s].label = label > 0.5 ? 1 : 0;

//		print("[%ld/%ld]: score = %lf, label = %d\n", s + 1, n_samples, scores[s].measure, scores[s].label);
	}
	qsort(scores, n_samples, sizeof(LabelledMeasure), cmp_labelledmeasure);

	double inv_n_pos, inv_n_neg;
	getInvPosNeg(samples, inv_n_pos, inv_n_neg);

	// Optimize the threshold as to have the minimum HTER
	double best_threshold = 0.5, best_HTER = 1.0, last_score = 0.0;
	long not_passed_pos = 0, not_passed_neg = 0;
	for (long s = 0; s < n_samples; s ++)
	{
		const double threshold = 0.5 * (scores[s].measure + last_score);
		const double crt_TAR = 1.0 - (double)not_passed_pos * inv_n_pos;
		const double crt_FAR = 1.0 - (double)not_passed_neg * inv_n_neg;
		const double crt_HTER = 0.5 * (crt_FAR + 1.0 - crt_TAR);

		if (crt_HTER < best_HTER)
		{
			best_HTER = crt_HTER;
			best_threshold = threshold;
		}

		last_score = scores[s].measure;
		if (scores[s].label == 1)
		{
			not_passed_pos ++;
		}
		else
		{
			not_passed_neg ++;
		}
	}

	// Set the threshold
	machine->setThreshold(best_threshold);

	// Cleanup
	delete[] scores;

//	{
//		double crt_TAR, crt_FAR, crt_HTER;
//		machine->setThreshold(best_threshold);
//		test(machine, samples, crt_TAR, crt_FAR, crt_HTER);
//		print("\tAFTER: TAR = %lf, FAR = %lf, HTER = %lf\n", crt_TAR, crt_FAR, crt_HTER);
//	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Computes the inverse of the number of positive and negative samples in a dataset

void LRTrainer::getInvPosNeg(DataSet* dataset, double& inv_n_pos, double& inv_n_neg)
{
	long n_neg = 0, n_pos = 0;
	for (long s = 0; s < dataset->getNoExamples(); s ++)
	{
		if (((const DoubleTensor*)dataset->getTarget(s))->get(0) > 0.5)
		{
			n_pos ++;
		}
		else
		{
			n_neg ++;
		}
	}

	inv_n_pos =  1.0 / (n_pos == 0 ? 1.0 : n_pos);
	inv_n_neg =  1.0 / (n_neg == 0 ? 1.0 : n_neg);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the gradient for the Grafting method (negative loglikelihoods + regularization terms)

void LRTrainer::getGradient(DataSet* dataset, double* gradients, double* buf_gradients,
				const double* weights, int size, double L1_prior, double L2_prior)
{
	// Initialize
	for (int i = 0; i <= size; i ++)
	{
		gradients[i] = 0.0;		// Accumulate from positive samples
		buf_gradients[i] = 0.0;		// Accumulate from negative samples
	}

	// Add the gradient of the -loglikelihood
	//	(normalize positive and negative samples separetely)
	for (long s = 0; s < dataset->getNoExamples(); s ++)
	{
		const DoubleTensor* example = (const DoubleTensor*)dataset->getExample(s);
		const double* data = (const double*)example->dataR();
		const double score = LRMachine::sigmoidEps(data, weights, size);

		const double label = ((const DoubleTensor*)dataset->getTarget(s))->get(0);
		double* dst = label > 0.5 ? gradients : buf_gradients;

		const double factor = - (label - score);
		for (int i = 0; i < size; i ++)
		{
			dst[i] += factor * data[i];
		}
		dst[size] += factor * 1.0;
	}

	double inv_n_pos, inv_n_neg;
	getInvPosNeg(dataset, inv_n_pos, inv_n_neg);

	for (int i = 0; i <= size; i ++)
	{
		gradients[i] = gradients[i] * inv_n_pos + buf_gradients[i] * inv_n_neg;
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
