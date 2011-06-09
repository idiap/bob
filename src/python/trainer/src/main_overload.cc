#include <boost/python.hpp>

using namespace boost::python;

void bind_trainer_kmeans_wrappers();
void bind_trainer_gmm_wrappers();

BOOST_PYTHON_MODULE(libpytorch_trainer_overload) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch classes and sub-classes for overloading trainers";
  
  bind_trainer_kmeans_wrappers();
  bind_trainer_gmm_wrappers();
}
