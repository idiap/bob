/**
 * @file python/config/src/main.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_error_eval();

BOOST_PYTHON_MODULE(libpytorch_error) {
  scope().attr("__doc__") = "Torch error classes and sub-classes";
  bind_error_eval();
}
