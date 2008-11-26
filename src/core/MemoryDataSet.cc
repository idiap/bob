#include "MemoryDataSet.h"
#include "Tensor.h"

namespace Torch {

MemoryDataSet::MemoryDataSet() : DataSet()
{
	examples = NULL;
}

MemoryDataSet::MemoryDataSet(int n_examples_, bool has_targets) : DataSet()
{
	n_examples = n_examples_;
	examples = NULL;
	examples = new TensorPair[n_examples];
}

TensorPair &MemoryDataSet::operator()(long t) const
{
   	if(examples == NULL) error("MemoryDataSet(): no examples in memory.");

	if((t < 0) || (t >= n_examples))
	{
   		error("MemoryDataSet(): example (%d) out-of-range [0-%d].", t, n_examples-1);
	}

	return examples[t];
}

void MemoryDataSet::cleanup()
{
	for (int i = 0; i < n_examples; i ++)
	{
		//if(examples[i].input != NULL) delete examples[i].input;
		//if(examples[i].target != NULL) delete examples[i].target;
	}
	delete [] examples;

	examples = NULL;
	n_examples = 0;
}

MemoryDataSet::~MemoryDataSet()
{
	cleanup();
}

}
