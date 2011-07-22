/**
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_core_mapstrings();
void bind_core_arraymapstrings();

BOOST_PYTHON_MODULE(libpytorch_core_mapstring) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch core classes and sub-classes for std::map manipulation with std::string keys from python";
  bind_core_mapstrings();
  bind_core_arraymapstrings();
}
