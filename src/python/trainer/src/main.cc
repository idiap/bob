#include <boost/python.hpp>

using namespace boost::python;

void bind_trainer_exception();
void bind_trainer_linear();
void bind_trainer_gmm();
void bind_trainer_kmeans();
void bind_trainer_rprop();
void bind_trainer_backprop();

BOOST_PYTHON_MODULE(libpytorch_trainer) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch classes and sub-classes for trainers";
  
  bind_trainer_exception();
  bind_trainer_linear();
  bind_trainer_gmm();
  bind_trainer_kmeans();
  bind_trainer_rprop();
  bind_trainer_backprop();
}
