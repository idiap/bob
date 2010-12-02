/**
 * @file src/core/python/src/DatasetXML.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the DatasetXML type into python
 */

#include <boost/python.hpp>

#include "core/DatasetXML.h"

using namespace boost::python;


void bind_core_DatasetXML()
{
	class_<Torch::core::DatasetXML, boost::shared_ptr<Torch::core::DatasetXML> >("DatasetXML","This class is used to represent a dataset in an XML format", init<char* >((arg("filename"))))
		;
}
