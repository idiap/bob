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
#include "core/array_type.h"

using namespace boost::python;

void bind_ndarray();
void bind_core_array_tinyvector();
void bind_core_array_range();
void bind_core_bz_numpy();
//void bind_core_array_examples(); ///< examples

# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  void bind_array_arith_bool_ ## D(); \
  void bind_array_arith_int8_ ## D(); \
  void bind_array_arith_int16_ ## D(); \
  void bind_array_arith_int32_ ## D(); \
  void bind_array_arith_int64_ ## D(); \
  void bind_array_arith_uint8_ ## D(); \
  void bind_array_arith_uint16_ ## D(); \
  void bind_array_arith_uint32_ ## D(); \
  void bind_array_arith_uint64_ ## D(); \
  void bind_array_arith_float32_ ## D(); \
  void bind_array_arith_float64_ ## D(); \
  void bind_array_arith_complex64_ ## D(); \
  void bind_array_arith_complex128_ ## D();
# include BOOST_PP_LOCAL_ITERATE()

void bind_array_base();
void bind_array_cast();
void bind_array_convert();
void bind_array_constructors();
void bind_array_indexing_1();
void bind_array_indexing_2();
void bind_array_indexing_3();
void bind_array_indexing_4();
void bind_array_info();

# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  void bind_array_math_bool_ ## D(); \
  void bind_array_math_int8_ ## D(); \
  void bind_array_math_int16_ ## D(); \
  void bind_array_math_int32_ ## D(); \
  void bind_array_math_int64_ ## D(); \
  void bind_array_math_uint8_ ## D(); \
  void bind_array_math_uint16_ ## D(); \
  void bind_array_math_uint32_ ## D(); \
  void bind_array_math_uint64_ ## D(); \
  void bind_array_math_float32_ ## D(); \
  void bind_array_math_float64_ ## D(); \
  void bind_array_math_complex64_ ## D(); \
  void bind_array_math_complex128_ ## D();
# include BOOST_PP_LOCAL_ITERATE()

void bind_array_memory();
void bind_array_order();
void bind_array_reductions_1();
void bind_array_reductions_2();
void bind_array_reductions_3();
void bind_array_reductions_4();
void bind_array_storage();
void bind_as_one_row();
void bind_array_cat();

BOOST_PYTHON_MODULE(libpytorch_core_array) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  bind_ndarray();
  scope().attr("__doc__") = "Torch core classes and sub-classes for array manipulation";
  
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
  //bind_core_array_examples(); ///< examples

  bind_core_array_tinyvector();
  bind_core_array_range();
  
  bind_array_base(); //this will create the class and has to come first!
  
# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  bind_array_arith_bool_ ## D(); \
  bind_array_arith_int8_ ## D(); \
  bind_array_arith_int16_ ## D(); \
  bind_array_arith_int32_ ## D(); \
  bind_array_arith_int64_ ## D(); \
  bind_array_arith_uint8_ ## D(); \
  bind_array_arith_uint16_ ## D(); \
  bind_array_arith_uint32_ ## D(); \
  bind_array_arith_uint64_ ## D(); \
  bind_array_arith_float32_ ## D(); \
  bind_array_arith_float64_ ## D(); \
  bind_array_arith_complex64_ ## D(); \
  bind_array_arith_complex128_ ## D();
# include BOOST_PP_LOCAL_ITERATE()

  bind_array_cast();
  bind_array_convert();
  bind_array_constructors();
  bind_array_indexing_1();
  bind_array_indexing_2();
  bind_array_indexing_3();
  bind_array_indexing_4();
  bind_array_info();

# define BOOST_PP_LOCAL_LIMITS (1, TORCH_MAX_DIM)
# define BOOST_PP_LOCAL_MACRO(D) \
  bind_array_math_bool_ ## D(); \
  bind_array_math_int8_ ## D(); \
  bind_array_math_int16_ ## D(); \
  bind_array_math_int32_ ## D(); \
  bind_array_math_int64_ ## D(); \
  bind_array_math_uint8_ ## D(); \
  bind_array_math_uint16_ ## D(); \
  bind_array_math_uint32_ ## D(); \
  bind_array_math_uint64_ ## D(); \
  bind_array_math_float32_ ## D(); \
  bind_array_math_float64_ ## D(); \
  bind_array_math_complex64_ ## D(); \
  bind_array_math_complex128_ ## D();
# include BOOST_PP_LOCAL_ITERATE()

  bind_array_memory();
  bind_array_order();
  bind_array_reductions_1();
  bind_array_reductions_2();
  bind_array_reductions_3();
  bind_array_reductions_4();
  bind_array_storage();
  bind_as_one_row();
  bind_array_cat();
}
