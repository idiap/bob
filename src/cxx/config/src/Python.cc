/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sun  6 Mar 18:48:52 2011 
 *
 * @brief Python interpreter initialization and destruction. 
 */

#include <Python.h>
#include "config/Python.h"

namespace conf = Torch::config;

boost::shared_ptr<conf::detail::Python> conf::detail::Python::s_instance;

conf::detail::Python::Python () {
  Py_Initialize(); //no-op if Python is already initialized
}

conf::detail::Python::~Python () {
  //this is called at the end of the program
  if (Py_IsInitialized()) Py_Finalize();
}

boost::shared_ptr<conf::detail::Python> conf::detail::Python::instance () {
  if (!conf::detail::Python::s_instance)
    s_instance.reset(new conf::detail::Python());
  return s_instance;
}
