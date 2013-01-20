/**
 * @file ip/python/zigzag.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the zigzag operation into python
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

#include "bob/core/python/ndarray.h"
#include "bob/ip/zigzag.h"

using namespace boost::python;

template <typename T>
static void inner_zigzag(bob::python::const_ndarray src, 
  blitz::Array<T,1>& dst, const bool rf) 
{
  bob::ip::zigzag(src.bz<T,2>(), dst, rf);
}

static object py_zigzag(bob::python::const_ndarray src, 
  object py_object, const bool rf=false)
{
  const bob::core::array::typeinfo& info = src.type();

  if(info.nd != 2) 
    PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support input array \
      with " SIZE_T_FMT " dimensions.", info.nd);

  extract<int> int_check(py_object);
  if(int_check.check()) 
  { //is int
    int n_coef = int_check();
    switch(info.dtype)
    {
      case bob::core::array::t_uint8:
        {
          bob::python::ndarray dst(bob::core::array::t_uint8, n_coef);
          blitz::Array<uint8_t,1> dst_ = dst.bz<uint8_t,1>();
          inner_zigzag<uint8_t>(src, dst_, rf);
          return dst.self();
        }
      case bob::core::array::t_uint16:
        {
          bob::python::ndarray dst(bob::core::array::t_uint16, n_coef);
          blitz::Array<uint16_t,1> dst_ = dst.bz<uint16_t,1>();
          inner_zigzag<uint16_t>(src, dst_, rf);
          return dst.self();
        }
      case bob::core::array::t_float64:
        {
          bob::python::ndarray dst(bob::core::array::t_float64, n_coef);
          blitz::Array<double,1> dst_ = dst.bz<double,1>();
          inner_zigzag<double>(src, dst_, rf);
          return dst.self();
        }
      default: 
        PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support array of type '%s'.", info.str().c_str());
    }
  }
  else
  {
    switch(info.dtype) 
    {
      case bob::core::array::t_uint8: 
      {
        blitz::Array<uint8_t,1> dst = extract<blitz::Array<uint8_t,1> >(py_object);
        inner_zigzag<uint8_t>(src, dst, rf);
        return object();
      }
      case bob::core::array::t_uint16: 
      {
        blitz::Array<uint16_t,1> dst = extract<blitz::Array<uint16_t,1> >(py_object);
        inner_zigzag<uint16_t>(src, dst, rf);
        return object();
      }
      case bob::core::array::t_float64: 
      {
        blitz::Array<double,1> dst = extract<blitz::Array<double,1> >(py_object);
        inner_zigzag<double>(src, dst, rf);
        return object();
      }
      default: 
        PYTHON_ERROR(TypeError, "bob.ip.zigzag() does not support array of type '%s'.", info.str().c_str());
    }
  }
}


BOOST_PYTHON_FUNCTION_OVERLOADS(zigzag_overloads, py_zigzag, 2, 3)

void bind_ip_zigzag() 
{
  def("zigzag", &py_zigzag,
    zigzag_overloads((arg("src"), arg("obj"), arg("right_first")=false),
    "Extracts a 1D NumPy array using a zigzag pattern from a 2D array/image. The second argument is\n 1. either the number of output coefficients to keep. In this case, an output 1D NumPy array is allocated and returned.\n 2. or a 1D NumPy array which will be updated with the zizag coefficients. In this case, nothing is returned (a None object)."));
}

