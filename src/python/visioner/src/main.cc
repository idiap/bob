/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu 21 Jul 2011 12:30:41 CEST
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_visioner_localize();

BOOST_PYTHON_MODULE(libpytorch_visioner) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch face localization bridge for visioner";
  bind_visioner_localize();
}
