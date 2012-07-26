/**
 * @file python/ip/src/zigzag.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the zigzag operation into python
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "core/python/ndarray.h"
#include "ip/zigzag.h"

using namespace boost::python;

template <typename T>
static void inner_zigzag(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const bool rf) 
{
  blitz::Array<T,1> dst_ = dst.bz<T,1>();
  bob::ip::zigzag(src.bz<T,2>(), dst_, rf);
}

static void py_zigzag_C(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const bool rf=false) 
{
  const bob::core::array::typeinfo& info = src.type();
  
  if(info.nd != 2) 
    PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support input array \
      with '%ld' dimensions.", info.nd);

  switch(info.dtype) 
  {
    case bob::core::array::t_uint8: 
      return inner_zigzag<uint8_t>(src, dst, rf);
    case bob::core::array::t_uint16: 
      return inner_zigzag<uint16_t>(src, dst, rf);
    case bob::core::array::t_float64: 
      return inner_zigzag<double>(src, dst, rf);
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support array of \
        type '%s'.", info.str().c_str());
  }
}

static object py_zigzag_P(bob::python::const_ndarray src, 
  const size_t n_coef, const bool rf=false)
{
  const bob::core::array::typeinfo& info = src.type();

  if(info.nd != 2) 
    PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support input array \
      with '%ld' dimensions.", info.nd);

  switch(info.dtype)
  {
    case bob::core::array::t_uint8:
      {
        bob::python::ndarray dst(bob::core::array::t_uint8, n_coef);
        blitz::Array<uint8_t,1> dst_ = dst.bz<uint8_t,1>();
        inner_zigzag<uint8_t>(src, dst, rf);
        return dst.self();
      }
    case bob::core::array::t_uint16:
      {
        bob::python::ndarray dst(bob::core::array::t_uint16, n_coef);
        blitz::Array<uint16_t,1> dst_ = dst.bz<uint16_t,1>();
        inner_zigzag<uint16_t>(src, dst, rf);
        return dst.self();
      }
    case bob::core::array::t_float64:
      {
        bob::python::ndarray dst(bob::core::array::t_float64, n_coef);
        blitz::Array<double,1> dst_ = dst.bz<double,1>();
        inner_zigzag<double>(src, dst, rf);
        return dst.self();
      }
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support array of \
        type '%s'.", info.str().c_str());
  }
}


BOOST_PYTHON_FUNCTION_OVERLOADS(zigzag_C_overloads, py_zigzag_C, 2, 3)
BOOST_PYTHON_FUNCTION_OVERLOADS(zigzag_P_overloads, py_zigzag_P, 2, 3)

void bind_ip_zigzag() 
{
  def("zigzag", &py_zigzag_C,
    zigzag_C_overloads((arg("src"), arg("dst"), arg("right_first")=false),
    "Extracts a 1D array using a zigzag pattern from a 2D array/image. \
      The number of coefficients to keep is given by the length of the dst \
      array."));
  def("zigzag", &py_zigzag_P,
    zigzag_P_overloads((arg("src"), arg("n_coefs"), arg("right_first")=false),
    "Extracts a 1D array using a zigzag pattern from a 2D array/image. This \
      function allocates and returns an array with n_coefs coefficients."));
}

