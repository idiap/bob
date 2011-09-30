/**
 * @author Francois Moulin <francois.moulin@idiap.ch> 
 * @author Laurent El-Shafey <lelshafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Fri 30 Sep 10:25:02 2011 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/pycore.h"

void bind_machine_exception();
void bind_machine_gmm();
void bind_machine_activation();
void bind_machine_linear();
void bind_machine_mlp();
void bind_machine_linear_scoring();
void bind_machine_ztnorm();
void bind_machine_jfa();

BOOST_PYTHON_MODULE(libpytorch_machine)
{
  Torch::python::setup_python("Torch classes and sub-classes for machine access");

  bind_machine_exception();
  bind_machine_gmm();
  bind_machine_activation();
  bind_machine_linear();
  bind_machine_mlp();
  bind_machine_linear_scoring();
  bind_machine_ztnorm();
  bind_machine_jfa();
}
