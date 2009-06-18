#include "MemoryDataSet.h"

using namespace Torch;

int main(int argc, char* argv[])
{
	const int n_tests = 1;
	const int n_max_examples = 10;
	const int n_targets = 3;
	const int frame_size = 4;
	const int n_max_frames = 60;
	//bool has_targets = true;
	bool has_targets = false;

	// Create some targets
	ShortTensor targets[n_targets];
	for (int i = 0; i < n_targets; i ++)
	{
		targets[i].resize(1);
		targets[i].fill(i);
	}

	MemoryDataSet mdataset;
	srand((unsigned int)time(0));

	// Do some random tests
	for (int t = 0; t < n_tests; t ++)
	{
		const int n_examples = 1 + rand() % n_max_examples;

		if(has_targets)
		{
			print("[%d/%d]: MemoryDataSet with [%d] examples of ShortTensor[%dx*] and [%d] targets ...\n",
				t + 1, n_tests, n_examples, frame_size, n_targets);

			// Resize
			mdataset.reset(n_examples, Tensor::Short, has_targets, Tensor::Short);
		}
		else
		{
			print("[%d/%d]: MemoryDataSet with [%d] examples of ShortTensor[%dx*] and no targets ...\n",
				t + 1, n_tests, n_examples, frame_size);

			// Resize
			mdataset.reset(n_examples, Tensor::Short);
		}

		CHECK_FATAL(mdataset.getNoExamples() == n_examples);

		// Fill the memory dataset with examples and targets
		for (int i = 0; i < n_examples; i ++)
		{
			Tensor* example = mdataset.getExample(i);
			CHECK_FATAL(example != 0);
			CHECK_FATAL(example->getDatatype() == Tensor::Short);
		
			const int n_frames = 1 + rand() % n_max_frames;

			example->resize(n_frames, frame_size);
			((ShortTensor*)example)->fill(i);

			if(has_targets)
			{
				Tensor* target = mdataset.getTarget(i);
				CHECK_FATAL(target == 0);
				mdataset.setTarget(i, &targets[i % n_targets]);
			}
		}

		// Retrieve examples and targets
		for (int i = 0; i < n_examples; i ++)
		{
			Tensor* example = mdataset.getExample(i);
			CHECK_FATAL(example != 0);
			CHECK_FATAL(example->getDatatype() == Tensor::Short);
			CHECK_FATAL(((ShortTensor*)example)->get(0, 0) == i);

			example->sprint("Sequence [%d]", i);

			if(has_targets)
			{
				Tensor* target = mdataset.getTarget(i);
				CHECK_FATAL(target != 0);
				CHECK_FATAL(target->getDatatype() == Tensor::Short);
				CHECK_FATAL(((ShortTensor*)target)->get(0) == i % n_targets);
			}
		}
	}

	print("\nOK\n");

	return 0;
}

