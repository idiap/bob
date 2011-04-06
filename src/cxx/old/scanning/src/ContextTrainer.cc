#include "scanning/ContextTrainer.h"
#include "scanning/ContextMachine.h"
#include "scanning/ContextDataSet.h"
#include "scanning/LRTrainer.h"
#include "core/MemoryDataSet.h"

static double getFARvsFRRRatio()
{
	return 0.3;
}

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ContextTrainer::ContextTrainer()
	:	m_validation_dataset(0)
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ContextTrainer::~ContextTrainer()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Set the validation dataset

bool ContextTrainer::setValidationData(DataSet* dataset)
{
	if (dataset == 0)
	{
		return false;
	}

	m_validation_dataset = dataset;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Build a dataset using the outputs of the context models

static bool buildCDataSet(MemoryDataSet& c_dataset, ContextDataSet* dataset, ContextMachine* ctx_machine)
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

		LRMachine& fmodel = ctx_machine->getFModel(f);

		for (long s = 0; s < dataset->getNoExamples(); s ++)
		{
			if (fmodel.forward(*dataset->getExample(s)) == false)
			{
				print("ContextTrainer::train - failed to run feature model [%d/%d]!\n", f + 1, NoFeatures);
				return false;
			}

			c_dataset.getExample(s)->set(f, fmodel.getOutput()(0) - fmodel.getThreshold());
		}
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Train the given machine on the given dataset

bool ContextTrainer::train()
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
		print("ContextTrainer::train - invalid parameters!\n");
		return false;
	}

	ContextMachine* ctx_machine = dynamic_cast<ContextMachine*>(m_machine);
	if (ctx_machine == 0)
	{
		print("ContextTrainer::train - can only train Context machines!\n");
		return false;
	}

	ContextDataSet* train_dataset = dynamic_cast<ContextDataSet*>(m_dataset);
	ContextDataSet* valid_dataset = dynamic_cast<ContextDataSet*>(m_validation_dataset);
	if (train_dataset == 0 || valid_dataset == 0)
	{
		print("ContextTrainer::train - can only use ContextDataSets!\n");
		return false;
	}

	const bool verbose = getBOption("verbose");

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Train the context feature models
	///////////////////////////////////////////////////////////////////////////////////////////////////

	LRTrainer lr_trainer;
	lr_trainer.setBOption("verbose", verbose);
	lr_trainer.setDOption("FARvsFRRRatio", getFARvsFRRRatio());
	lr_trainer.setBOption("useL1", false);

	for (int f = 0; f < NoFeatures; f ++)
	{
		train_dataset->reset(f);
		valid_dataset->reset(f);

		if (verbose == true)
		{
			print("ContextTrainer::train - LR training the feature model [%d/%d] ...\n", f + 1, NoFeatures);
		}

		if (	lr_trainer.setMachine(&ctx_machine->getFModel(f)) == false ||
			lr_trainer.setData(train_dataset) == false ||
			lr_trainer.setValidationData(valid_dataset) == false ||
			lr_trainer.train() == false)
		{
			print("ContextTrainer::train - failed to train feature model [%d/%d]!\n", f + 1, NoFeatures);
			return false;
		}
	}

	///////////////////////////////////////////////////////////////////////////////////////////////////
	// Train the combined classifier using the outputs of the context feature models
	///////////////////////////////////////////////////////////////////////////////////////////////////

	// Build datasets using the outputs of the context feature models (train & validation)
	if (verbose == true)
	{
		print("ContextTrainer::train - building dataset to train the combined classifier...\n");
	}

	const long n_train_samples = train_dataset->getNoExamples();
	const long n_valid_samples = valid_dataset->getNoExamples();

	MemoryDataSet c_train_dataset(n_train_samples, Tensor::Double, true, Tensor::Double);
	buildCDataSet(c_train_dataset, train_dataset, ctx_machine);

	MemoryDataSet c_valid_dataset(n_valid_samples, Tensor::Double, true, Tensor::Double);
	buildCDataSet(c_valid_dataset, valid_dataset, ctx_machine);

	// Train the combined classifier using these datasets
	lr_trainer.setDOption("FARvsFRRRatio", getFARvsFRRRatio());
	lr_trainer.setBOption("useL1", true);

	if (verbose == true)
	{
		print("ContextTrainer::train - LR training the combined classifier ...\n");
	}

	if (	lr_trainer.setMachine(&ctx_machine->getCModel()) == false ||
		lr_trainer.setData(&c_train_dataset) == false ||
		lr_trainer.setValidationData(&c_valid_dataset) == false ||
		lr_trainer.train() == false)
	{
		print("ContextTrainer::train - failed to train the combined classifier!\n");
		return false;
	}

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Test the Context machine

void ContextTrainer::test(ContextMachine* machine, ContextDataSet* samples, double& TAR, double& FAR, double& HTER)
{
	const double threshold = 0.5;
	DoubleTensor buf_context;

	long passed_pos = 0, passed_neg = 0;
	long cnt_pos = 0, cnt_neg = 0;
	for (long s = 0; s < samples->getNoExamples(); s ++)
	{
		samples->getContext(s)->copyTo(buf_context);
		CHECK_FATAL(machine->forward(buf_context) == true);

		const double label = ((const DoubleTensor*)samples->getTarget(s))->get(0);

		if (machine->isPattern())
		{
			long* dst = label > threshold ? &passed_pos : &passed_neg;
			(*dst) ++;
		}

		if (label >= threshold)
		{
			cnt_pos ++;
		}
		else
		{
			cnt_neg ++;
		}
	}

	TAR = (double)passed_pos / (cnt_pos == 0 ? 1.0 : (cnt_pos + 0.0));
	FAR = (double)passed_neg / (cnt_neg == 0 ? 1.0 : (cnt_neg + 0.0));
	const double FRR = 1.0 - TAR;
	HTER = (getFARvsFRRRatio() * FAR + FRR) / (getFARvsFRRRatio() + 1.0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

}
