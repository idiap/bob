/**
 * @file io/python/exception.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief io exceptions
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

#include <bob/io/Exception.h>
#include <bob/core/python/exception.h>

using namespace bob::python;

void bind_io_exception() {
  register_exception_translator<bob::io::IndexError>(PyExc_IndexError);
  register_exception_translator<bob::io::TypeError>(PyExc_TypeError);
  register_exception_translator<bob::io::UnsupportedTypeError>(PyExc_TypeError);
  register_exception_translator<bob::io::FileNotReadable>(PyExc_IOError);
  register_exception_translator<bob::io::ImageUnsupportedType>(PyExc_TypeError);
}
