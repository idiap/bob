/**
 * @file core/python/profile.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the Google profiler into python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/config.h>
#include <boost/python.hpp>

#if WITH_PERFTOOLS
#include <google/profiler.h>
#endif

using namespace boost::python;

void bind_core_profiler()
{
#if WITH_PERFTOOLS
  def("ProfilerStart", &ProfilerStart);
  def("ProfilerStop", &ProfilerStop);
  def("ProfilerFlush", &ProfilerFlush);
#endif
}
