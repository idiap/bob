/**
 * @author Francois Moulin <francois.moulin@idiap.ch> 
 * @author Laurent El-Shafey <lelshafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Fri 30 Sep 10:25:02 2011 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/pycore.h"

void bind_trainer_kmeans_wrappers();
void bind_trainer_gmm_wrappers();

BOOST_PYTHON_MODULE(libpytorch_trainer_overload) {

  Torch::python::setup_python("Torch classes and sub-classes for overloading trainers");
  
  bind_trainer_kmeans_wrappers();
  bind_trainer_gmm_wrappers();
}
