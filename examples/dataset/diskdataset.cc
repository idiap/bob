#include "DiskDataSet.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	srand((unsigned int)time(0));

	const int n_max_examples = 1000;
	const int width = 320;
	const int height = 240;
	ShortTensor timage(height, width);

	// Create some targets
	const int n_targets = 3;
	ShortTensor targets[n_targets];
	for (int i = 0; i < n_targets; i ++)
	{
		targets[i].resize(1);
		targets[i].fill(i);
	}

	// Generate some image images (tensors) and store them in files
	const int n_files = 3;
	const char* filenames[n_files] =
	{
		"data1.tensor",
		"data2.tensor",
		"data3.tensor"
	};

	int n_examples = 0;
	for (int i = 0; i < n_files; i ++)
	{
		TensorFile tf;
		CHECK_FATAL(tf.openWrite(filenames[i], Tensor::Short, 2, height, width) == true);

		const int n_file_examples = rand() % n_max_examples;
		print("Writing [%d] images[%dx%d] into file [%s] ...\n",
			n_file_examples, width, height, filenames[i]);

		for (int j = 0; j < n_file_examples; j ++)
		{
			timage.fill(n_examples + j);
			CHECK_FATAL(tf.save(timage) == true);
		}
		n_examples += n_file_examples;
	}

	print("\nOK\n");

	// Load the files into some DiskDataSet
	DiskDataSet ddataset(Tensor::Short);
	for (int i = 0; i < n_files; i ++)
	{
		CHECK_FATAL(ddataset.load(filenames[i]) == true);
		print("Loading [%s] - [%d] examples so far.\n",
			filenames[i], ddataset.getNoExamples());
	}
	CHECK_FATAL(ddataset.getNoExamples() == n_examples);

	print("\nOK\n");

	// Assign some targets
	print("Assigning [%d] targets ...\n", n_examples);
	for (int i = 0; i < n_examples; i ++)
	{
		Tensor* target = ddataset.getTarget(i);
		CHECK_FATAL(target == 0);
		ddataset.setTarget(i, &targets[i % n_targets]);
	}

	print("\nOK\n");

	// Read the examples and targets
	print("Reading [%d] examples and targets ...\n", n_examples);
	for (int i = 0; i < n_examples; i ++)
	{
		Tensor* example = ddataset.getExample(i);
		CHECK_FATAL(example != 0);
		CHECK_FATAL(example->getDatatype() == Tensor::Short);
		CHECK_FATAL(((ShortTensor*)example)->get(0, 0) == i);

		Tensor* target = ddataset.getTarget(i);
		CHECK_FATAL(target != 0);
		CHECK_FATAL(target->getDatatype() == Tensor::Short);
		CHECK_FATAL(((ShortTensor*)target)->get(0) == i % n_targets);
	}

	print("\nOK\n");

   	return 0;
}

