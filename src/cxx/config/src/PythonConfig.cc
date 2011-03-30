/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat  5 Mar 20:26:56 2011 
 *
 * @brief Implementation of the readout/saving of configuration in Python
 */

#include <boost/filesystem.hpp>
#include "config/PythonConfig.h"
#include "config/Python.h"
#include "config/Exception.h"

namespace conf = Torch::config;
namespace bp = boost::python;

void conf::detail::pyload(const boost::filesystem::path& path, bp::dict& dict) {
  boost::shared_ptr<conf::detail::Python> 
    interpreter = conf::detail::Python::instance();
  try {
    bp::object module = bp::import("__main__");
    bp::object name_space = module.attr("__dict__");
    name_space["__file__"] = bp::str(path.string().c_str());
    bp::exec_file(path.string().c_str(), name_space, dict);
  }
  catch (bp::error_already_set) {
    /**
      PyObject *ptype, *pvalue, *ptraceback;
      PyErr_Fetch(&ptype, &pvalue, &ptraceback); //this will clear the error

      bp::object exception_type(bp::handle<>(ptype));
      bp::object exception_value(bp::handle<>(pvalue));
      bp::object traceback(bp::handle<>(ptraceback));

      str exception_typename = bp::extract<str>(exception_type.attr("__name__"));

      throw conf::ParserError(bp::extrac<std::string>(exception_typename), 
      bp::extract<std::string>(pvalue));
     **/
    PyErr_Print();
    throw conf::PythonError();
  }
}

void conf::detail::pysave(const boost::filesystem::path& path, const bp::dict& dict) {
  boost::shared_ptr<conf::detail::Python> 
    interpreter = conf::detail::Python::instance();
  throw conf::NotImplemented("save", path.extension());
}
