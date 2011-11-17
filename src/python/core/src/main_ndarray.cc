/**
 * @file main_array.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/ndarray.h"

void bind_core_bz_numpy();
void bind_core_ndarray_numpy();
void bind_core_array_tinyvector();
void bind_core_array_typeinfo();
//void bind_core_array_examples(); ///< examples

BOOST_PYTHON_MODULE(libpytorch_core_ndarray) {
  Torch::python::setup_python("Torch core classes and sub-classes for array manipulation");

  bind_core_bz_numpy();
  bind_core_ndarray_numpy();
  bind_core_array_tinyvector();
  bind_core_array_typeinfo();
  //bind_core_array_examples(); ///< examples
}
