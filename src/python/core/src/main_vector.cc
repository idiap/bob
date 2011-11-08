/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 30 Sep 12:18:20 2011 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/ndarray.h"

void bind_core_vectors();
void bind_core_arrayvectors_1();
void bind_core_arrayvectors_2();
void bind_core_arrayvectors_3();
void bind_core_arrayvectors_4();

BOOST_PYTHON_MODULE(libpytorch_core_vector) {

  Torch::python::setup_python("Torch core classes and sub-classes for std::vector manipulation from python");

  bind_core_vectors();
  bind_core_arrayvectors_1();
  bind_core_arrayvectors_2();
  bind_core_arrayvectors_3();
  bind_core_arrayvectors_4();
}
