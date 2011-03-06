/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun  6 Mar 18:48:52 2011 
 *
 * @brief Python interpreter initialization and destruction. 
 */

#include <Python.h>
#include "config/Python.h"

namespace conf = Torch::config;

boost::shared_ptr<conf::Python> conf::Python::s_instance;

conf::Python::Python () {
  Py_Initialize();
}

conf::Python::~Python () {
  Py_Finalize();
}

boost::shared_ptr<conf::Python> conf::Python::instance () {
  if (!conf::Python::s_instance) s_instance.reset(new conf::Python());
  return s_instance;
}
