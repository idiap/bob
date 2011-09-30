/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 30 Sep 10:32:54 2011 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/pycore.h"

void bind_measure_error();

BOOST_PYTHON_MODULE(libpytorch_measure) {

  Torch::python::setup_python("Torch error measure classes and sub-classes");

  bind_measure_error();
}
