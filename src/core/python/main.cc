/**
 * @file core/python/main.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/config.h>
#include <bob/python/ndarray.h>

using namespace boost::python;

void bind_core_version();
void bind_core_exception();
void bind_core_logging();
void bind_core_bz_numpy();
void bind_core_ndarray_numpy();
void bind_core_typeinfo();
void bind_core_convert();
void bind_core_tinyvector();
void bind_core_numpy_scalars();

#if WITH_PERFTOOLS
void bind_core_profiler();
#endif

BOOST_PYTHON_MODULE(_core) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob core classes and sub-classes");

  bind_core_version();
  bind_core_exception();
  bind_core_logging();
  bind_core_bz_numpy();
  bind_core_ndarray_numpy();
  bind_core_typeinfo();
  bind_core_convert();
  bind_core_tinyvector();
  bind_core_numpy_scalars();

#if WITH_PERFTOOLS
  bind_core_profiler();
#endif
}
