#include "DataSet.h"

namespace Torch {

DataSet::DataSet(Tensor::Type example_type_)
	: 	m_size(0),
		m_example_type(example_type_) 
{
}

DataSet::~DataSet()
{
}

}
