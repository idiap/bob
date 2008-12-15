#include "MemoryDataSet.h"
#include "Tensor.h"

namespace Torch {

MemoryDataSet::MemoryDataSet(Tensor::Type example_type_, Tensor::Type target_type_)
	: DataSet(example_type_, target_type_)
{
	examples = NULL;
	targets = NULL;
}

MemoryDataSet::MemoryDataSet(int n_examples_, Tensor::Type example_type_, bool has_targets, Tensor::Type target_type_)
	: DataSet(example_type_, target_type_)
{
	n_examples = n_examples_;
	examples = NULL;
	examples = new Tensor* [n_examples];
	if(example_type == Tensor::Double)
		for(int i = 0 ; i < n_examples ; i++)
			examples[i] = new DoubleTensor;
	else if(example_type == Tensor::Int)
		for(int i = 0 ; i < n_examples ; i++)
			examples[i] = new IntTensor;
	else error("MemoryDataSet: sorry example type not supported yet");

	targets = NULL;
	if(has_targets)
	{
		targets = new Tensor* [n_examples];
		if(target_type == Tensor::Short)
			for(int i = 0 ; i < n_examples ; i++) targets[i] = new ShortTensor;
		else error("MemoryDataSet: sorry target type not supported yet");
	}
}

Tensor* MemoryDataSet::getExample(long t)
{
   	if(examples == NULL) error("MemoryDataSet(): no examples in memory.");

	if((t < 0) || (t >= n_examples))
	{
   		error("MemoryDataSet(): example (%d) out-of-range [0-%d].", t, n_examples-1);
	}

	return examples[t];
}

Tensor &MemoryDataSet::operator()(long t)
{
   	if(examples == NULL) error("MemoryDataSet(): no examples in memory.");

	if((t < 0) || (t >= n_examples))
	{
   		error("MemoryDataSet(): example (%d) out-of-range [0-%d].", t, n_examples-1);
	}

	return *(examples[t]);
}

Tensor* MemoryDataSet::getTarget(long t)
{
   	if(targets == NULL) error("MemoryDataSet(): no targets in memory.");

	if((t < 0) || (t >= n_examples))
	{
   		error("MemoryDataSet(): target (%d) out-of-range [0-%d].", t, n_examples-1);
	}

	return targets[t];
}

void MemoryDataSet::cleanup()
{
	if(examples != NULL)
	{
	   	for(int i = 0 ; i < n_examples ; i++) delete examples[i];
		delete [] examples;
	}
	if(targets != NULL)
	{
	   	for(int i = 0 ; i < n_examples ; i++) delete targets[i];
		delete [] targets;
	}

	examples = NULL;
	targets = NULL;
	n_examples = 0;
}

MemoryDataSet::~MemoryDataSet()
{
	cleanup();
}

}
