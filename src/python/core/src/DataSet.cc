/**
 * @file src/python/core/src/DataSet.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 *
 * @brief Binds the Dataset object type into python
 */

#include <boost/python.hpp>

#include "core/DataSet.h"

using namespace boost::python;

void bind_core_DataSet()
{
	class_<Torch::DataSet, boost::shared_ptr<Torch::DataSet>, bases<Torch::Object>, boost::noncopyable>("Machine", "", no_init)
		// .def("getExample", &Torch::DataSet::getExample, (arg("self"), arg("index")), "get example from dataset")
		;
}
