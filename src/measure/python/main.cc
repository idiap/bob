/**
 * @file measure/python/main.cc
 * @date Wed Apr 20 08:19:36 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/python/ndarray.h"

void bind_measure_error();

BOOST_PYTHON_MODULE(_measure) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob error measure classes and sub-classes");

  bind_measure_error();
}
