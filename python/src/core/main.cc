/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

void bind_core_object();
void bind_tensor_object();

BOOST_PYTHON_MODULE(libpytorch_core) {
  bind_core_object();
  bind_tensor_object();
}
