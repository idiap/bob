#include "DataSet.h"

namespace Torch {

DataSet::DataSet(Tensor::Type example_type_, Tensor::Type target_type_) : example_type(example_type_), target_type(target_type_)
{
	n_examples = 0;
}

DataSet::~DataSet()
{
}

}
