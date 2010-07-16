/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_trainer_gtfile();

BOOST_PYTHON_MODULE(libpytorch_trainer) {
  scope().attr("__doc__") = "Torch trainer classes and sub-classes";
  bind_trainer_gtfile();
}
