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
		.def("load",(void (Torch::core::XMLParser::*)(const char*, Torch::core::Dataset&))&Torch::core::XMLParser::load, (arg("self"), arg("filename"), arg("dataset")), "Load function")
		;
}

//static void load(const char *filename, Dataset& dataset);
