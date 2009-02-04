#include "DataSet.h"

namespace Torch {

DataSet::DataSet(Tensor::Type example_type_, Tensor::Type target_type_)
	: 	m_n_examples(0),
		m_example_type(example_type_), m_target_type(target_type_)
{
}

DataSet::~DataSet()
{
}

}
