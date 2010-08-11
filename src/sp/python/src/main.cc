/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_sp_spcore();
void bind_sp_transforms();

BOOST_PYTHON_MODULE(libpytorch_sp) {
  scope().attr("__doc__") = "Torch signal processing classes and sub-classes";
  bind_sp_spcore();
  bind_sp_transforms();
}
