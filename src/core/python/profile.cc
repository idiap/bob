/**
 * @file core/python/profile.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds the Google profiler into python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "bob/config.h"
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
