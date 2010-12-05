/**
 * @file src/core/python/src/XMLParser.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the XMLParser into python
 */

#include <boost/python.hpp>

#include "core/XMLParser.h"

using namespace boost::python;


void bind_core_XMLParser()
{
	class_<Torch::core::XMLParser, boost::shared_ptr<Torch::core::XMLParser> >("XMLParser","This class is used to parse an XML file", init<>())
		.def("load",(Torch::core::Dataset* (Torch::core::XMLParser::*)(const char*))&Torch::core::XMLParser::load, return_value_policy<manage_new_object>(), (arg("self"), arg("filename")), "Load function")
		;
}

