/**
 * @file src/python/math/src/main.cc 
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_math_linsolve();

BOOST_PYTHON_MODULE(libpytorch_math) {
  scope().attr("__doc__") = "Torch mathematical classes and sub-classes";
  bind_math_linsolve();
}
