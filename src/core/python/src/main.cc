/**
 * @file main.cc 
 * @author <a href="mailto:andre.dos.anjos@cern.ch">Andre Anjos</a> 
 *
 * @brief Combines all modules to make up the complete bindings
 */

#include <boost/python.hpp>

using namespace boost::python;

void bind_core_object();
void bind_core_file();
void bind_core_tensor();
void bind_core_tensorfile();
void bind_core_profiler();
void bind_core_DataSet();
void bind_core_ListDataSet();

BOOST_PYTHON_MODULE(libpytorch_core) {
  scope().attr("__doc__") = "Torch core classes and sub-classes";
  bind_core_object();
  bind_core_file();
  bind_core_tensor();
  bind_core_tensorfile();
  bind_core_profiler();
  bind_core_DataSet();
  bind_core_ListDataSet();
}
