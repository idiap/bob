/**
 * @file python/config/src/main.cc 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_config_exception();
void bind_config_configuration();

BOOST_PYTHON_MODULE(libpytorch_config) {
  scope().attr("__doc__") = "Torch configuration classes and sub-classes";
  bind_config_exception();
  bind_config_configuration();
}
