/**
 * @file main_array.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>
#include <boost/python/numeric.hpp>
#define PY_ARRAY_UNIQUE_SYMBOL torch_NUMPY_ARRAY_API
#include <numpy/arrayobject.h>
#include <dlfcn.h>
#include <iostream>

using namespace boost::python;

void bind_core_bz_numpy();
void bind_core_array_examples(); ///< examples

BOOST_PYTHON_MODULE(libpytorch_core_array2) {

  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Automatic to-from python for blitz::Array<> as NumPy arrays";
  
  // Gets the current dlopenflags and save it
  PyThreadState *tstate = PyThreadState_GET();
  if(!tstate)
    throw std::runtime_error("Can not get python dlopenflags.");
  int old_value = tstate->interp->dlopenflags;

  // Unsets the RTLD_GLOBAL flag
  tstate->interp->dlopenflags = old_value & (~RTLD_GLOBAL);
  // Loads numpy with the RTLD_GLOBAL flag unset
  import_array();
  // Resets the RTLD_GLOBAL flag
  tstate->interp->dlopenflags = old_value;

  //Sets the boost::python::numeric::array interface to use numpy.ndarray
  //as basis. This is not strictly required, but good to set as a baseline.
  numeric::array::set_module_and_type("numpy", "ndarray");

  bind_core_bz_numpy();
  bind_core_array_examples(); ///< examples
}
