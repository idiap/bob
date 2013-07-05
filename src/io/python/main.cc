/**
 * @file io/python/main.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Combines all modules to make up the complete bindings
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

#include <bob/config.h>
#include <bob/python/ndarray.h>
#include <boost/python.hpp>
#include <bob/io/CodecRegistry.h>

static bool get_ignore_double_registration() {
  return bob::io::CodecRegistry::instance()->ignoreDoubleRegistration();
}

static void set_ignore_double_registration(bool v) {
  bob::io::CodecRegistry::instance()->ignoreDoubleRegistration(v);
}

void bind_io_version();
void bind_io_file();
void bind_io_hdf5();
void bind_io_hdf5_extras();
void bind_io_datetime();

#if WITH_FFMPEG
void bind_io_video();
#endif

BOOST_PYTHON_MODULE(_io) {
  boost::python::docstring_options docopt(true, true, false);
  bob::python::setup_python("bob classes and sub-classes for io access");

  bind_io_version();
  bind_io_file();
  bind_io_hdf5();
  bind_io_hdf5_extras();
  bind_io_datetime();

#if WITH_FFMPEG
  bind_io_video();
#endif

  boost::python::def("__get_ignore_double_registration__", &get_ignore_double_registration);
  boost::python::def("__set_ignore_double_registration__", &set_ignore_double_registration);
}
