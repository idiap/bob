/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 21 Jul 2011 12:30:41 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/pycore.h"

void bind_visioner_localize();

BOOST_PYTHON_MODULE(libpytorch_visioner) {
  
  Torch::python::setup_python("Torch face localization bridge for visioner");

  bind_visioner_localize();
}
