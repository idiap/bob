/**
 * @file src/core/python/src/profile.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Binds the Tensor object type into python 
 */

#include <boost/python.hpp>

#ifdef HAS_GOOGLE_PERFTOOLS
#include <google/profiler.h>
#endif

using namespace boost::python;

void bind_core_profiler()
{
#ifdef HAS_GOOGLE_PERFTOOLS
  def("ProfilerStart", &ProfilerStart);
  def("ProfilerStop", &ProfilerStop);
  def("ProfilerFlush", &ProfilerFlush);
#endif
}
