/**
 * @file src/core/tensor.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 * @author <a href="mailto:niklas.johansson@idiap.ch">Niklas Johansson</a>
 *
 * @brief Binds the Tensor object type into python
 */

#include <boost/python.hpp>

#include "core/ListDataSet.h"

using namespace boost::python;

/*
static Torch::Tensor* getExample(Torch::ListDataSet &self, const long index)
{
	return NULL;
	// return self.getExample(index);
}
*/

static int load(Torch::ListDataSet &self, const char *filename)
{
	return self.load(filename);
}

void bind_core_ListDataSet()
{
	class_<Torch::ListDataSet, boost::shared_ptr<Torch::ListDataSet>, bases<Torch::DataSet> >("ListDataSet")
		.def("load", &load, (arg("self"), arg("filename")), "test function")
		;
}
