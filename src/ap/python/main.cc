/**
 * @file ap/python/main.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <boost/python.hpp>
#include <bob/python/ndarray.h>

void bind_ap_ceps();

BOOST_PYTHON_MODULE(_ap)
{
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob audio processing classes and sub-classes");

  bind_ap_ceps();
}
