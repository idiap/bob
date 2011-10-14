#include <boost/python.hpp>

using namespace boost::python;

void bind_machine_exception();
void bind_machine_gmm();
void bind_machine_activation();
void bind_machine_linear();
void bind_machine_mlp();
void bind_machine_linear_scoring();
void bind_machine_ztnorm();
void bind_machine_jfa();
void bind_machine_plda();
void bind_machine_wiener();

BOOST_PYTHON_MODULE(libpytorch_machine)
{
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch classes and sub-classes for machine access";

  bind_machine_exception();
  bind_machine_gmm();
  bind_machine_activation();
  bind_machine_linear();
  bind_machine_mlp();
  bind_machine_linear_scoring();
  bind_machine_ztnorm();
  bind_machine_jfa();
  bind_machine_plda();
  bind_machine_wiener();
}
