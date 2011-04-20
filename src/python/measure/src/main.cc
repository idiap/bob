/**
 * @file python/config/src/main.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_measure_error();

BOOST_PYTHON_MODULE(libpytorch_measure) {
  scope().attr("__doc__") = "Torch error measure classes and sub-classes";
  bind_measure_error();
}
