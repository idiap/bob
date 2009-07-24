#include "ProfileTrainer.h"
#include "ProfileMachine.h"
#include "ProfileDataSet.h"
#include "LRTrainer.h"
#include "MemoryDataSet.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ProfileTrainer::ProfileTrainer()
	:	m_validation_dataset(0)
{
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
// Build a dataset using the outputs of the profile models

static bool buildCDataSet(MemoryDataSet& c_dataset, ProfileDataSet* dataset, ProfileMachine* pf_machine)
{
	// Set targets
	for (long s = 0; s < dataset->getNoExamples(); s ++)
	{
		c_dataset.getExample(s)->resize(NoFeatures);
		c_dataset.setTarget(s, dataset->getTarget(s));
	}

	// Set the output for each feature
	for (int f = 0; f < NoFeatures; f ++)
	{
		dataset->reset(f);

		LRMachine& fmodel = pf_machine->getFModel(f);
		const double* score = (const double*)fmodel.getOutput().dataR();
		const double threshold = fmodel.getThreshold();

		for (long s = 0; s < dataset->getNoExamples(); s ++)
		{
			if (fmodel.forward(*dataset->getExample(s)) == false)
			{
				print("ProfileTrainer::train - failed to run feature model [%d/%d]!\n", f + 1, NoFeatures);
				return false;
			}

			c_dataset.getExample(s)->set(f, *score >= threshold ? 1.0 : -1.0);
		}
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Train the given machine on the given dataset

bool ProfileTrainer::train()
{
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

	const bool verbose = getBOption("verbose");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Train the profile feature models
	///////////////////////////////////////////////////////////////////////////////////////////////////

	LRTrainer lr_trainer;
	lr_trainer.setBOption("verbose", verbose);

	for (int f = 0; f < NoFeatures; f ++)
	{
		train_dataset->reset(f);
		valid_dataset->reset(f);

		if (verbose == true)
		{
			print("ProfileTrainer::train - LR training the feature model [%d/%d] ...\n", f + 1, NoFeatures);
		}

		if (	lr_trainer.setMachine(&pf_machine->getFModel(f)) == false ||
			lr_trainer.setData(train_dataset) == false ||
			lr_trainer.setValidationData(valid_dataset) == false ||
			lr_trainer.train() == false)
		{
			print("ProfileTrainer::train - failed to train feature model [%d/%d]!\n", f + 1, NoFeatures);
			return false;
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Train the combined classifier using the outputs of the profile feature models
	///////////////////////////////////////////////////////////////////////////////////////////////////

	// Build datasets using the outputs of the profile feature models (train & validation)
	if (verbose == true)
	{
		print("ProfileTrainer::train - building dataset to train the combined classifier...\n");
	}

	const long n_train_samples = train_dataset->getNoExamples();
	const long n_valid_samples = valid_dataset->getNoExamples();

	MemoryDataSet c_train_dataset(n_train_samples, Tensor::Double, true, Tensor::Double);
	buildCDataSet(c_train_dataset, train_dataset, pf_machine);

	MemoryDataSet c_valid_dataset(n_valid_samples, Tensor::Double, true, Tensor::Double);
	buildCDataSet(c_valid_dataset, valid_dataset, pf_machine);

	// Train the combined classifier using these datasets
	if (verbose == true)
	{
		print("ProfileTrainer::train - LR training the combined classifier ...\n");
	}

	if (	lr_trainer.setMachine(&pf_machine->getCModel()) == false ||
		lr_trainer.setData(&c_train_dataset) == false ||
		lr_trainer.setValidationData(&c_valid_dataset) == false ||
		lr_trainer.train() == false)
	{
		print("ProfileTrainer::train - failed to train the combined classifier!\n");
		return false;
	}

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Test the Profile machine (returns the detection rate in percentages)

double ProfileTrainer::test(ProfileMachine* machine, ProfileDataSet* samples)
{
	long correct = 0;
	for (long s = 0; s < samples->getNoExamples(); s ++)
	{
		const Profile* profile = samples->getProfile(s);
		static DoubleTensor buf_profile;
		profile->copyTo(buf_profile);

		CHECK_FATAL(machine->forward(buf_profile) == true);

		if (	(((DoubleTensor*)samples->getTarget(s))->get(0) >= 0.5) ==
			(machine->isPattern()))
		{
			correct ++;
		}
	}

	return 100.0 * (correct + 0.0) / (samples->getNoExamples() == 0 ? 1.0 : samples->getNoExamples());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Test the Profile machine (returns the TAR and FAR and FA)

void ProfileTrainer::test(ProfileMachine* machine, ProfileDataSet* samples, double& tar, double& far, long& fa)
{
	tar = 0.0;
	far = 0.0;
	fa = 0;

	long cnt_tar = 0;
	long cnt_far = 0;
	long cnt_pos = 0;
	long cnt_neg = 0;
	for (long s = 0; s < samples->getNoExamples(); s ++)
	{
		const bool valid = ((DoubleTensor*)samples->getTarget(s))->get(0) >= 0.5;

		static DoubleTensor buf_profile;
		samples->getProfile(s)->copyTo(buf_profile);
		CHECK_FATAL(machine->forward(buf_profile) == true);

		if (machine->isPattern() == true)
		{
			if (valid == true)
				cnt_tar ++;
			else
				cnt_far ++;
		}

		if (valid == false)
			cnt_neg ++;
		else
			cnt_pos ++;
	}

	tar = (cnt_tar + 0.0) / (cnt_pos == 0 ? 1.0 : (cnt_pos + 0.0));
	far = (cnt_far + 0.0) / (cnt_neg == 0 ? 1.0 : (cnt_neg + 0.0));
	fa = cnt_far;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
