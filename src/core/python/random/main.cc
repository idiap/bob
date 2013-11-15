/**
 * @file core/python/random/main.cc
 * @date Mon Jul 11 18:31:22 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief boost::random bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/python/ndarray.h"

void bind_core_random();

BOOST_PYTHON_MODULE(_core_random) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob core classes and sub-classes for accessing boost::random objects from python");
  bind_core_random();
}
