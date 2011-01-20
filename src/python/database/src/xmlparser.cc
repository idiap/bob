/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the XMLParser into python
 */

#include <boost/python.hpp>

#include "core/XMLParser.h"

using namespace boost::python;
namespace db = Torch::core;

void bind_database_xmlparser()
{
	class_<db::XMLParser, boost::shared_ptr<db::XMLParser> >("XMLParser", "This class is used to parse an XML file", init<>())
		.def("load",(void (db::XMLParser::*)(const char*, db::Dataset&))&db::XMLParser::load, (arg("self"), arg("filename"), arg("dataset")), "Load function")
		;
}

