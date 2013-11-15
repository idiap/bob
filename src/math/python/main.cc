/**
 * @file math/python/main.cc
 * @date Sat Mar 19 19:49:51 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "bob/python/ndarray.h"

void bind_math_lp_interiorpoint();
void bind_math_linsolve();
void bind_math_norminv();
void bind_math_stats();
void bind_math_histogram();
void bind_math_pavx();

BOOST_PYTHON_MODULE(_math) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob mathematical classes and sub-classes");

  bind_math_lp_interiorpoint();
  bind_math_linsolve();
  bind_math_norminv();
  bind_math_stats();
  bind_math_histogram();
  bind_math_pavx();
}
