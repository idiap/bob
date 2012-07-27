/**
 * @file python/ip/src/flipflop.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the flip and flop operations into python
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
#include "ip/flipflop.h"

using namespace boost::python;

template <typename T, int N>
static void inner_flip(bob::python::const_ndarray src, 
  bob::python::ndarray dst) 
{
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  bob::ip::flip<T>(src.bz<T,N>(), dst_);
}

template <int N>
static void inner_flip_type(bob::python::const_ndarray src, 
  bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8:
      return inner_flip<uint8_t,N>(src, dst);
    case bob::core::array::t_uint16:
      return inner_flip<uint16_t,N>(src, dst);
    case bob::core::array::t_float64:
      return inner_flip<double,N>(src, dst);
    default:
      PYTHON_ERROR(TypeError, 
        "image flipping does not support array of type '%s'.", 
        info.str().c_str());
  }
}

static void py_flip_c(bob::python::const_ndarray src, 
  bob::python::ndarray dst)
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_flip_type<2>(src, dst);
    case 3: return inner_flip_type<3>(src, dst);
    default:
      PYTHON_ERROR(TypeError, 
        "image flipping does not support array of '%ld' dimensions.", 
        info.nd);
  }
}

static object py_flip_p(bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.nd) {
    case 2:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], info.shape[1]);
        inner_flip_type<2>(src, dst);
        return dst.self();
      }
    case 3:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], info.shape[1], 
          info.shape[2]);
        inner_flip_type<3>(src, dst);
        return dst.self();
      }
    default:
      PYTHON_ERROR(TypeError, 
        "image flipping does not support array of '%ld' dimensions.", 
        info.nd);
  }
}


template <typename T, int N>
static void inner_flop(bob::python::const_ndarray src, 
  bob::python::ndarray dst) 
{
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  bob::ip::flop<T>(src.bz<T,N>(), dst_);
}

template <int N>
static void inner_flop_type(bob::python::const_ndarray src, 
  bob::python::ndarray dst) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8:
      return inner_flop<uint8_t,N>(src, dst);
    case bob::core::array::t_uint16:
      return inner_flop<uint16_t,N>(src, dst);
    case bob::core::array::t_float64:
      return inner_flop<double,N>(src, dst);
    default:
      PYTHON_ERROR(TypeError, 
        "image flopping does not support array of type '%s'.", 
        info.str().c_str());
  }
}

static void py_flop_c(bob::python::const_ndarray src, 
  bob::python::ndarray dst)
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_flop_type<2>(src, dst);
    case 3: return inner_flop_type<3>(src, dst);
    default:
      PYTHON_ERROR(TypeError, 
        "image flopping does not support array of '%ld' dimensions.", 
        info.nd);
  }
}

static object py_flop_p(bob::python::const_ndarray src) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.nd) {
    case 2:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], info.shape[1]);
        inner_flop_type<2>(src, dst);
        return dst.self();
      }
    case 3:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], info.shape[1], 
          info.shape[2]);
        inner_flop_type<3>(src, dst);
        return dst.self();
      }
    default:
      PYTHON_ERROR(TypeError, 
        "image flopping does not support array of '%ld' dimensions.", 
        info.nd);
  }
}

void bind_ip_flipflop() 
{
  static const char* FLIP_DOC = "Flip a 2D or 3D array/image upside-down. The destination array should have the same size and type as the source array.";
  static const char* FLIP_P_DOC = "Flip a 2D or 3D array/image upside-down. The output array is allocated and returned.";
  static const char* FLOP_DOC = "Flop a 2D or 3D array/image left-right. The destination array should have the same size and type as the source array.";
  static const char* FLOP_P_DOC = "Flop a 2D or 3D array/image left-right. The output array is allocated and returned.";

  def("flip", &py_flip_c, (arg("src"), arg("dst")), FLIP_DOC); 
  def("flip", &py_flip_p, (arg("src")), FLIP_P_DOC); 
  def("flop", &py_flop_c, (arg("src"), arg("dst")), FLOP_DOC); 
  def("flop", &py_flop_p, (arg("src")), FLOP_P_DOC); 
}
