/**
 * @file main_array.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_ndarray();
void bind_core_array_tinyvector();
void bind_core_array_range();
void bind_array_arithmetics_1();
void bind_array_arithmetics_2();
void bind_array_arithmetics_3();
void bind_array_arithmetics_4();
void bind_array_base();
void bind_array_cast();
void bind_array_convert();
void bind_array_constructors();
void bind_array_indexing_1();
void bind_array_indexing_2();
void bind_array_indexing_3();
void bind_array_indexing_4();
void bind_array_info();
void bind_array_math_1();
void bind_array_math_2();
void bind_array_math_3();
void bind_array_math_4();
void bind_array_memory();
void bind_array_order();
void bind_array_reductions_1();
void bind_array_reductions_2();
void bind_array_reductions_3();
void bind_array_reductions_4();
void bind_array_storage();
void bind_as_one_row();

BOOST_PYTHON_MODULE(libpytorch_core_array) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  bind_ndarray();
  scope().attr("__doc__") = "Torch core classes and sub-classes for array manipulation";
  bind_core_array_tinyvector();
  bind_core_array_range();
  
  bind_array_base(); //this will create the class and has to come first!
  bind_array_arithmetics_1();
  bind_array_arithmetics_2();
  bind_array_arithmetics_3();
  bind_array_arithmetics_4();
  bind_array_cast();
  bind_array_convert();
  bind_array_constructors();
  bind_array_indexing_1();
  bind_array_indexing_2();
  bind_array_indexing_3();
  bind_array_indexing_4();
  bind_array_info();
  bind_array_math_1();
  bind_array_math_2();
  bind_array_math_3();
  bind_array_math_4();
  bind_array_memory();
  bind_array_order();
  bind_array_reductions_1();
  bind_array_reductions_2();
  bind_array_reductions_3();
  bind_array_reductions_4();
  bind_array_storage();
  bind_as_one_row();
}
