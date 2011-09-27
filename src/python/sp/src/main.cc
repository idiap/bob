/**
 * @file src/python/sp/src/main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_sp_spcore();
void bind_sp_convolution();
void bind_sp_extrapolate();
void bind_sp_fft_dct();

BOOST_PYTHON_MODULE(libpytorch_sp) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch signal processing classes and sub-classes";
  bind_sp_spcore();
  bind_sp_convolution();
  bind_sp_extrapolate();
  bind_sp_fft_dct();
}
