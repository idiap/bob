/**
 * @file main_array.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#include <iostream>

using namespace boost::python;

void bind_core_bz_numpy();

BOOST_PYTHON_MODULE(libpytorch_core_array2) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Automatic to-from python for blitz::Array<> as NumPy arrays";
  bind_core_bz_numpy();
}
