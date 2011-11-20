/**
 * @author Francois Moulin <francois.moulin@idiap.ch> 
 * @author Laurent El-Shafey <lelshafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Fri 30 Sep 10:25:02 2011 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/ndarray.h"

void bind_trainer_linear();
void bind_trainer_gmm();
void bind_trainer_kmeans();
void bind_trainer_rprop();
void bind_trainer_backprop();
void bind_trainer_jfa();

BOOST_PYTHON_MODULE(libpytorch_trainer) {

  Torch::python::setup_python("Torch classes and sub-classes for trainers");
  
  bind_trainer_linear();
  bind_trainer_gmm();
  bind_trainer_kmeans();
  bind_trainer_rprop();
  bind_trainer_backprop();
  bind_trainer_jfa();
}
