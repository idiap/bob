/**
 * @file src/python/math/src/main.cc 
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_math_eig();
void bind_math_linear();
void bind_math_linsolve();
void bind_math_lu_det();
void bind_math_norminv();
void bind_math_svd();
void bind_math_stats();

BOOST_PYTHON_MODULE(libpytorch_math) {
  docstring_options docopt; 
# if !defined(TORCH_DEBUG)
  docopt.disable_cpp_signatures();
# endif
  scope().attr("__doc__") = "Torch mathematical classes and sub-classes";
  bind_math_eig();
  bind_math_linear();
  bind_math_linsolve();
  bind_math_lu_det();
  bind_math_norminv();
  bind_math_svd();
  bind_math_stats();
}
