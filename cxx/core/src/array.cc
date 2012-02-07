/**
 * @file cxx/core/src/array.cc
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Some buffer stuff
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include <boost/format.hpp>
#include "core/array.h"

namespace ca = bob::core::array;

ca::typeinfo::typeinfo():
  dtype(ca::t_unknown),
  nd(0)
{
}

ca::typeinfo::typeinfo(const ca::typeinfo& other): 
  dtype(other.dtype)
{
  set_shape(other.nd, other.shape);
}

ca::typeinfo& ca::typeinfo::operator= (const ca::typeinfo& other) {
  dtype = other.dtype;
  set_shape(other.nd, other.shape);
  return *this;
}

void ca::typeinfo::reset() {
  dtype = ca::t_unknown;
  nd = 0;
}

bool ca::typeinfo::is_valid() const {
  return (dtype != ca::t_unknown) && (nd > 0) && (nd <= (BOB_MAX_DIM+1)) && has_valid_shape();
}

void ca::typeinfo::update_strides() {
  switch (nd) {
    case 0:
      return;
    case 1:
      stride[0] = 1;
      return;
    case 2:
      stride[1] = 1;
      stride[0] = shape[1];
      return;
    case 3:
      stride[2] = 1;
      stride[1] = shape[2];
      stride[0] = shape[1]*shape[2];
      return;
    case 4:
      stride[3] = 1;
      stride[2] = shape[3];
      stride[1] = shape[2]*shape[3];
      stride[0] = shape[1]*shape[2]*shape[3];
      return;
    case 5:
      stride[4] = 1;
      stride[3] = shape[4];
      stride[2] = shape[3]*shape[4];
      stride[1] = shape[2]*shape[3]*shape[4];
      stride[0] = shape[1]*shape[2]*shape[3]*shape[4];
      return;
    default:
      break;
  }
  throw std::invalid_argument("unsupported number of dimensions");
}

size_t ca::typeinfo::size() const {
  size_t retval = 1;
  for (size_t k=0; k<nd; ++k) retval *= shape[k];
  return retval;
}

size_t ca::typeinfo::buffer_size() const {
  return size()*ca::getElementSize(dtype);
}

static bool same_shape(size_t nd, const size_t* s1, const size_t* s2) {
  for (size_t k=0; k<nd; ++k) if (s1[k] != s2[k]) return false;
  return true;
}

bool ca::typeinfo::is_compatible(const ca::typeinfo& other) const {
  return (dtype == other.dtype) && (nd == other.nd) && same_shape(nd, shape, other.shape);
}

std::string ca::typeinfo::str() const {
  boost::format s("%dD, %s (%d bytes), %d bytes");
  s % nd % item_str() % item_size() % buffer_size();
  return s.str();
}

void ca::typeinfo::reset_shape() {
  shape[0] = 0;
}

bool ca::typeinfo::has_valid_shape() const {
  return shape[0] != 0;
}
