/**
 * @file main_array.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/pycore.h"

void bind_core_bz_numpy();
//void bind_core_array_examples(); ///< examples

BOOST_PYTHON_MODULE(libpytorch_core_array) {
  Torch::python::setup_python("Torch core classes and sub-classes for array manipulation");

  bind_core_bz_numpy();
  //bind_core_array_examples(); ///< examples
}
