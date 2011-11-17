/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include "core/python/ndarray.h"

void bind_math_eig();
void bind_math_linsolve();
void bind_math_lu_det();
void bind_math_norminv();
void bind_math_svd();
void bind_math_stats();

BOOST_PYTHON_MODULE(libpytorch_math) {

  Torch::python::setup_python("Torch mathematical classes and sub-classes");

  bind_math_eig();
  bind_math_linsolve();
  bind_math_lu_det();
  bind_math_norminv();
  bind_math_svd();
  bind_math_stats();
}
