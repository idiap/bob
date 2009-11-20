#include "torch5spro.h"

using namespace Torch;

int main(int argc, char* argv[])
{
	const long n_examples = 1000000;
	const int n_targets = 10;
	const int size0 = 19;
	const int size1 = 19;
	bool has_targets = true;

	// Create some targets
	ShortTensor targets[n_targets];
	for (int i = 0; i < n_targets; i ++)
	{
		targets[i].resize(n_targets);
		targets[i].fill(0);
		targets[i].set(i, 1);
	}

	MemoryDataSet mdataset;

	print("MemoryDataSet with [%d] examples of ShortTensor[%dx%d] and [%d] targets ...\n", n_examples, size0, size1, n_targets);

	// Resize
	mdataset.reset(n_examples, Tensor::Short, has_targets, Tensor::Short);

	CHECK_FATAL(mdataset.getNoExamples() == n_examples);

	// Fill the memory dataset with examples and targets
	for (long i = 0; i < n_examples; i ++)
	{
		Tensor* example = mdataset.getExample(i);
		CHECK_FATAL(example != 0);
		CHECK_FATAL(example->getDatatype() == Tensor::Short);
		example->resize(size0, size1);
		((ShortTensor*)example)->fill(i%1024);

		if(has_targets)
		{
			Tensor* target = mdataset.getTarget(i);
			CHECK_FATAL(target == 0);
			mdataset.setTarget(i, &targets[i % n_targets]);
		}
	}

	print("Memory allocation done.\n");
	
	print("Reading memory ...\n");
	
	// Retrieve examples and targets
	for (long i = 0; i < n_examples; i ++)
	{
		Tensor* example = mdataset.getExample(i);
		CHECK_FATAL(example != 0);
		CHECK_FATAL(example->getDatatype() == Tensor::Short);
		CHECK_FATAL(((ShortTensor*)example)->get(0, 0) == i % 1024);

		if(has_targets)
		{
			Tensor* target = mdataset.getTarget(i);
			CHECK_FATAL(target != 0);
			CHECK_FATAL(target->getDatatype() == Tensor::Short);
			//CHECK_FATAL(((ShortTensor*)target)->get(i % n_targets) == i % n_targets);
			CHECK_FATAL(((ShortTensor*)target)->get(i % n_targets) == 1);
		}
	}

	print("\nOK\n");

	return 0;
}

