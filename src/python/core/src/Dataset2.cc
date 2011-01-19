/**
 * @file src/core/python/src/Dataset2.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Dataset type into python
 */

#include <boost/python.hpp>

#include "core/Dataset2.h"

using namespace boost::python;


void bind_core_Dataset()
{
	class_<Torch::core::Dataset, boost::shared_ptr<Torch::core::Dataset> >("Dataset","This class is used to represent a dataset", init<>())
		;
}
