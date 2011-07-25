/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 11 Jul 2011 16:42:45 CEST
 *
 * @brief boost::random bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_core_tuple();

BOOST_PYTHON_MODULE(libpytorch_core_tuple) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch core classes and sub-classes for accessing boost::tuple objects from python";
  bind_core_tuple();
}
