/**
 * @file core/cxx/array.cc
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Some buffer stuff
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

#include <boost/format.hpp>
#include <bob/core/array.h>

bob::core::array::typeinfo::typeinfo():
  dtype(bob::core::array::t_unknown),
  nd(0)
{
}

bob::core::array::typeinfo::typeinfo(const bob::core::array::typeinfo& other): 
  dtype(other.dtype)
{
  set_shape(other.nd, other.shape);
}

bob::core::array::typeinfo& bob::core::array::typeinfo::operator= (const bob::core::array::typeinfo& other) {
  dtype = other.dtype;
  set_shape(other.nd, other.shape);
  return *this;
}

void bob::core::array::typeinfo::reset() {
  dtype = bob::core::array::t_unknown;
  nd = 0;
}

bool bob::core::array::typeinfo::is_valid() const {
  return (dtype != bob::core::array::t_unknown) && (nd > 0) && (nd <= (BOB_MAX_DIM+1)) && has_valid_shape();
}

void bob::core::array::typeinfo::update_strides() {
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
  throw std::runtime_error("unsupported number of dimensions");
}

size_t bob::core::array::typeinfo::size() const {
  size_t retval = 1;
  for (size_t k=0; k<nd; ++k) retval *= shape[k];
  return retval;
}

size_t bob::core::array::typeinfo::buffer_size() const {
  return size()*bob::core::array::getElementSize(dtype);
}

static bool same_shape(size_t nd, const size_t* s1, const size_t* s2) {
  for (size_t k=0; k<nd; ++k) if (s1[k] != s2[k]) return false;
  return true;
}

bool bob::core::array::typeinfo::is_compatible(const bob::core::array::typeinfo& other) const {
  return (dtype == other.dtype) && (nd == other.nd) && same_shape(nd, shape, other.shape);
}

std::string bob::core::array::typeinfo::str() const {
  boost::format s("dtype: %s (%d); shape: [%s]; size: %d bytes");
  size_t sz = 0;
  size_t buf_sz = 0;
  if (dtype != bob::core::array::t_unknown) {
    //otherwise it throws
    sz = item_size();
    buf_sz = buffer_size();
  }
  s % item_str() % sz; 
  switch (nd) {
    case 0:
      s % "";
      break;
    case 1:
      s % (boost::format("%d") % shape[0]).str();
      break;
    case 2:
      s % (boost::format("%d,%d") % shape[0] % shape[1]).str();
      break;
    case 3:
      s % (boost::format("%d,%d,%d") % shape[0] % shape[1] % shape[2]).str();
      break;
    case 4:
      s % (boost::format("%d,%d,%d,%d") % shape[0] % shape[1] % shape[2] % shape[3]).str();
      break;
    default:
      s % ">4 dimensions?";
      break;
  }
  s % buf_sz;
  return s.str();
}

void bob::core::array::typeinfo::reset_shape() {
  shape[0] = 0;
}

bool bob::core::array::typeinfo::has_valid_shape() const {
  return shape[0] != 0;
}
