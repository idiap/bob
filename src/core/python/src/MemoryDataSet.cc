/**
 * @file src/core/python/src/MemoryDataSet.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 *
 * @brief Binds the MemoryDataSet type into python
 */

#include <boost/python.hpp>

#include "core/DataSet.h"

using namespace boost::python;

void bind_core_DataSet()
{
	class_<Torch::MemoryDataSet, boost::shared_ptr<Torch::MemoryDataSet>, bases<Torch::DataSet>, boost::noncopyable>("Machine", "", no_init)

		.def("getExample", &Torch::DataSet::getExample, (arg("self"), arg("index")), "get example from dataset")
		;
}
