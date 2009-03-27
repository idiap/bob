#include "ProfileTrainer.h"
#include "ProfileMachine.h"
#include "ProfileDataSet.h"
#include "FLDATrainer.h"
#include "LRTrainer.h"
#include "MemoryDataSet.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ProfileTrainer::ProfileTrainer()
	:	m_validation_dataset(0)
{
	addFOption("FMinTAR", 0.85f, "Minimum TAR for some profile feature model");
        addFOption("FMinTRR", 0.85f, "Minimum TRR for some profile feature model");
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ProfileTrainer::~ProfileTrainer()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Set the validation dataset

bool ProfileTrainer::setValidationData(DataSet* dataset)
{
	if (dataset == 0)
	{
		return false;
	}

	m_validation_dataset = dataset;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Check if some target in the dataset is positive

static bool isPosTarget(DataSet& dataset, long s)
{
	return ((const DoubleTensor*)dataset.getTarget(s))->get(0) > 0.1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Computes the scores for some dataset

template <typename Model>
static bool test(Model& model, DataSet& dataset, double* neg_scores, double* pos_scores)
{
	const double* score = (const double*)model.getOutput().dataR();

	// Test each sample
	int i_neg = 0, i_pos = 0;
	const int n_samples = dataset.getNoExamples();
	for (long s = 0; s < n_samples; s ++)
	{
		if (model.forward(*dataset.getExample(s)) == false)
		{
			return false;
		}

		if (isPosTarget(dataset, s) == false)
		{
			neg_scores[i_neg ++] = *score;
		}
		else
		{
			pos_scores[i_pos ++] = *score;
		}
	}

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Computes the TAR and TRR for some dataset

template <typename Model>
static bool test(Model& model, DataSet& dataset, int& cnt_TAR, int& cnt_TRR)
{
	cnt_TAR = cnt_TRR = 0;

	const double* score = (const double*)model.getOutput().dataR();

	// Test each sample
	const int n_samples = dataset.getNoExamples();
	for (long s = 0; s < n_samples; s ++)
	{
		if (model.forward(*dataset.getExample(s)) == false)
		{
			return false;
		}

		if (isPosTarget(dataset, s) == false)
		{
			if (*score < model.getThreshold())
			{
				cnt_TRR ++;
			}
		}
		else
		{
			if (*score >= model.getThreshold())
			{
				cnt_TAR ++;
			}
		}

		//const DoubleTensor* target = (const DoubleTensor*)dataset.getTarget(s);
		//const DoubleTensor* example = (const DoubleTensor*)dataset.getExample(s);

//		const int n_selected = example->size(0);
//		for (int i = 0; i < n_selected; i ++)
//		{
//			print("%d ", (int)example->get(i));
//		}
//
//		print("=== [%d] vs. [%f]\n", (int)target->get(0), *score);
	}

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Get some threshold as to have the required minimum TAR and TRR

static double tune(	double* neg_scores, int n_neg_scores,
			double* pos_scores, int n_pos_scores,
			double min_TAR, double min_TRR)
{
	qsort(neg_scores, n_neg_scores, sizeof(double), compare_doubles);
	qsort(pos_scores, n_pos_scores, sizeof(double), compare_doubles);

	const int idx_TAR = getInRange(FixI((1.0 - min_TAR) * n_pos_scores), 0, n_pos_scores - 1);
	const int idx_TRR = getInRange(FixI(min_TRR * n_neg_scores), 0, n_neg_scores - 1);

	return 0.5 * (pos_scores[idx_TAR] + neg_scores[idx_TRR]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Train the given machine on the given dataset

bool ProfileTrainer::train()
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
		print("ProfileTrainer::train - invalid parameters!\n");
		return false;
	}
	ProfileMachine* pf_machine = dynamic_cast<ProfileMachine*>(m_machine);
	if (pf_machine == 0)
	{
		print("ProfileTrainer::train - can only train Profile machines!\n");
		return false;
	}

	ProfileDataSet* train_dataset = dynamic_cast<ProfileDataSet*>(m_dataset);
	ProfileDataSet* valid_dataset = dynamic_cast<ProfileDataSet*>(m_validation_dataset);
	if (train_dataset == 0 || valid_dataset == 0)
	{
		print("ProfileTrainer::train - can only use ProfileDataSets!\n");
		return false;
	}

	//const int n_train_samples = train_dataset->getNoExamples();
	const int n_valid_samples = valid_dataset->getNoExamples();

	const bool verbose = getBOption("verbose");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Train the profile feature models on the training dataset
	///////////////////////////////////////////////////////////////////////////////////////////////////

	FLDATrainer flda_trainer;
	for (int f = 0; f < NoFeatures; f ++)
	{
		train_dataset->reset(f);

		if (	flda_trainer.setMachine(&pf_machine->getFModel(f)) == false ||
			flda_trainer.setData(train_dataset) == false ||
			flda_trainer.train() == false)
		{
			print("ProfileTrainer::train - failed to train some feature model!\n");
			return false;
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Select the best profile feature models on the validation dataset
	///////////////////////////////////////////////////////////////////////////////////////////////////

	// Get the number of positive and negative examples in the validation dataset
	int n_neg = 0, n_pos = 0;
	for (long s = 0; s < n_valid_samples; s ++)
	{
		if (isPosTarget(*valid_dataset, s) == false)
		{
			n_neg ++;
		}
		else
		{
			n_pos ++;
		}
	}
	CHECK_FATAL(n_neg + n_pos == n_valid_samples);

	double* neg_scores = new double[n_neg];
	double* pos_scores = new double[n_pos];

	const float min_f_TAR = getFOption("FMinTAR");
	const float min_f_TRR = getFOption("FMinTRR");

	// Select the best features as to have the required TAR and TRR
	for (int f = 0; f < NoFeatures; f ++)
	{
		valid_dataset->reset(f);

		// Compute the scores and the threshold
		FLDAMachine& fmodel = pf_machine->getFModel(f);
		test(fmodel, *valid_dataset, neg_scores, pos_scores);
		fmodel.setThreshold(tune(neg_scores, n_neg, pos_scores, n_pos, min_f_TAR, min_f_TRR));

		// Test again with the new threshold to select the best features
		int cnt_TAR, cnt_TRR;
		test(fmodel, *valid_dataset, cnt_TAR, cnt_TRR);
		pf_machine->setFSelected(f,	cnt_TAR >= FixI(min_f_TAR * n_pos) &&
						cnt_TRR >= FixI(min_f_TRR * n_neg));

		if (verbose == true)
		{
			const double inv_pos = 1.0 / (n_pos + 0.0);
			const double inv_neg = 1.0 / (n_neg + 0.0);

			print("%s[%s]: TAR = %5.3f%%, TRR = %5.3f%%\n",
				pf_machine->isFSelected(f) == true ? "*" : " ", FeatureNames[f],
				100.0 * inv_pos * cnt_TAR, 100.0 * inv_neg * cnt_TRR);
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Train the combined classifier using the outputs of the profile feature models
	///////////////////////////////////////////////////////////////////////////////////////////////////

	DoubleTensor pos_target(1), neg_target(1);
	pos_target.fill(1.0);
	neg_target.fill(0.0);

	int n_selected = 0;
	for (int f = 0; f < NoFeatures; f ++)
		if (pf_machine->isFSelected(f) == true)
		{
			n_selected ++;
		}
	if (n_selected == 0)
	{
		return false;
	}

	// Build a memory dataset to contain the outputs of the profile feature models
	MemoryDataSet lr_dataset(n_valid_samples, Tensor::Double, true, Tensor::Double);

	for (long s = 0; s < n_valid_samples; s ++)
	{
		lr_dataset.getExample(s)->resize(n_selected);

		const DoubleTensor* target = (const DoubleTensor*)valid_dataset->getTarget(s);
		lr_dataset.setTarget(s, target->get(0) < 0.1 ? &neg_target : &pos_target);
	}

	int index = 0;
	for (int f = 0; f < NoFeatures; f ++)
		if (pf_machine->isFSelected(f) == true)
		{
			valid_dataset->reset(f);

			FLDAMachine& fmodel = pf_machine->getFModel(f);
			const double* score = (const double*)fmodel.getOutput().dataR();
			const double threshold = fmodel.getThreshold();

			for (long s = 0; s < n_valid_samples; s ++)
			{
				if (fmodel.forward(*valid_dataset->getExample(s)) == false)
				{
					print("ProfileTrainer::train - failed to run some feature model!\n");
					return false;
				}

				lr_dataset.getExample(s)->set(index, *score >= threshold ? 1.0 : -1.0);
			}

			index ++;
		}

//	print("Training data:\n");
//	for (long s = 0; s < n_valid_samples; s ++)
//	{
//		const DoubleTensor* target = (const DoubleTensor*)lr_dataset.getTarget(s);
//		const DoubleTensor* example = (const DoubleTensor*)lr_dataset.getExample(s);
//
//		for (int i = 0; i < n_selected; i ++)
//		{
//			print("%d ", (int)example->get(i));
//		}
//
//		print("=== [%d]\n", (int)target->get(0));
//	}

	// Train the LR combined classifier
	LRTrainer lr_trainer;
	if (	lr_trainer.setMachine(&pf_machine->getCModel()) == false ||
		lr_trainer.setData(&lr_dataset) == false ||
		lr_trainer.train() == false)
	{
		print("ProfileTrainer::train - failed to train the combined classifier!\n");
		return false;
	}

	// Test the LR combined classifier
	int cnt_TAR, cnt_TRR;
	test(pf_machine->getCModel(), lr_dataset, cnt_TAR, cnt_TRR);

	if (verbose == true)
	{
		const double inv_pos = 1.0 / (n_pos + 0.0);
		const double inv_neg = 1.0 / (n_neg + 0.0);

		print("--------------------------------------------------------------------------\n");
		print(">>> TAR = %5.3f%%, TRR = %5.3f%%\n",
				100.0 * inv_pos * cnt_TAR, 100.0 * inv_neg * cnt_TRR);
	}

	// Cleanup
	delete[] neg_scores;
	delete[] pos_scores;

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
