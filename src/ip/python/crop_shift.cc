/**
 * @file python/ip/src/crop_shift.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @Sun 20 Nov 17:13:34 2011 CET
 * @brief Binds crop and shift operations into python
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

#include "bob/core/python/ndarray.h"
#include "bob/ip/crop.h"
#include "bob/ip/shift.h"

using namespace boost::python;

template <typename T, int N>
static void inner_crop1(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const int y, const int x, const size_t h, 
  const size_t w, const bool allow_out, const bool zero_out) 
{
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  bob::ip::crop<T>(src.bz<T,N>(), dst_, y, x, h, w, allow_out, zero_out);
}

template <int N>
static void inner_crop1_type(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const int y, const int x, const size_t h,
  const size_t w, const bool allow_out, const bool zero_out)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.dtype) {
    case bob::core::array::t_uint8:
      return inner_crop1<uint8_t,N>(src, dst, y, x, h, w, allow_out, 
                                    zero_out);
    case bob::core::array::t_uint16:
      return inner_crop1<uint16_t,N>(src, dst, y, x, h, w, allow_out, 
                                     zero_out);
    case bob::core::array::t_float64:
      return inner_crop1<double,N>(src, dst, y, x, h, w, allow_out,
                                   zero_out);
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.crop() does not support array of type '%s'.", 
        info.str().c_str());
  }
}

static void py_crop1_c(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const int y, const int x, const size_t h, 
  const size_t w, const bool allow_out=false, const bool zero_out=false)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd) {
    case 2: 
      return inner_crop1_type<2>(src, dst, y, x, h, w, allow_out, zero_out);
    case 3: 
      return inner_crop1_type<3>(src, dst, y, x, h, w, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.crop() does not support array with \
        " SIZE_T_FMT " dimensions.", info.nd);
  }
}

static object py_crop1_p(bob::python::const_ndarray src, const int y, 
  const int x, const size_t h, const size_t w, const bool allow_out=false, 
  const bool zero_out=false) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd) {
    case 2:
      {
        bob::python::ndarray dst(info.dtype, h, w);
        inner_crop1_type<2>(src, dst, y, x, h, w, allow_out, zero_out);
        return dst.self();
      }
    case 3:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], h, w); 
        inner_crop1_type<3>(src, dst, y, x, h, w, allow_out, zero_out);
        return dst.self();
      }
    default:
      PYTHON_ERROR(TypeError, "bob.ip.crop() does not support array with \
        " SIZE_T_FMT " dimensions.", info.nd);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(py_crop1_c_overloads, py_crop1_c, 6, 8)
BOOST_PYTHON_FUNCTION_OVERLOADS(py_crop1_p_overloads, py_crop1_p, 5, 7)


template <typename T, int N>
static void inner_crop2(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst, 
  bob::python::ndarray dmask, const int y, const int x, const size_t h, 
  const size_t w, const bool allow_out, const bool zero_out) 
{
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  bob::ip::crop<T>(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, y, x, h, 
                   w, allow_out, zero_out);
}

template <int N>
static void inner_crop2_type(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst,
  bob::python::ndarray dmask, const int y, const int x, const size_t h,
  const size_t w, const bool allow_out, const bool zero_out)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.dtype) {
    case bob::core::array::t_uint8:
      return inner_crop2<uint8_t,N>(src, smask, dst, dmask, y, x, h, w, 
                                    allow_out, zero_out);
    case bob::core::array::t_uint16:
      return inner_crop2<uint16_t,N>(src, smask, dst, dmask, y, x, h, w, 
                                    allow_out, zero_out);
    case bob::core::array::t_float64:
      return inner_crop2<double,N>(src, smask, dst, dmask, y, x, h, w, 
                                    allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.crop() does not support array of type '%s'.", 
        info.str().c_str());
  }
}

static void py_crop2_c(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst,
  bob::python::ndarray dmask, const int y, const int x, const size_t h, 
  const size_t w, const bool allow_out=false, const bool zero_out=false)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd) {
    case 2: 
      return inner_crop2_type<2>(src, smask, dst, dmask, y, x, h, w, 
                                 allow_out, zero_out);
    case 3: 
      return inner_crop2_type<3>(src, smask, dst, dmask, y, x, h, w, 
                                 allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.crop() does not support array with \
        " SIZE_T_FMT " dimensions.", info.nd);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(py_crop2_c_overloads, py_crop2_c, 8, 10)


template <typename T, int N>
static void inner_shift1(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const int y, const int x, const bool allow_out, 
  const bool zero_out) 
{
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  bob::ip::shift<T>(src.bz<T,N>(), dst_, y, x, allow_out, zero_out);
}

template <int N>
static void inner_shift1_type(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const int y, const int x, const bool allow_out,
  const bool zero_out)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.dtype) {
    case bob::core::array::t_uint8:
      return inner_shift1<uint8_t,N>(src, dst, y, x, allow_out, zero_out);
    case bob::core::array::t_uint16:
      return inner_shift1<uint16_t,N>(src, dst, y, x, allow_out, zero_out);
    case bob::core::array::t_float64:
      return inner_shift1<double,N>(src, dst, y, x, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.shift() does not support array of type '%s'.", 
        info.str().c_str());
  }
}

static void py_shift1_c(bob::python::const_ndarray src, 
  bob::python::ndarray dst, const int y, const int x, 
  const bool allow_out=false, const bool zero_out=false)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd) {
    case 2: 
      return inner_shift1_type<2>(src, dst, y, x, allow_out, zero_out);
    case 3: 
      return inner_shift1_type<3>(src, dst, y, x, allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shift() does not support array with \
        " SIZE_T_FMT " dimensions.", info.nd);
  }
}

static object py_shift1_p(bob::python::const_ndarray src, const int y, 
  const int x, const bool allow_out=false, const bool zero_out=false) 
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd) {
    case 2:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], info.shape[1]);
        inner_shift1_type<2>(src, dst, y, x, allow_out, zero_out);
        return dst.self();
      }
    case 3:
      {
        bob::python::ndarray dst(info.dtype, info.shape[0], info.shape[1], 
          info.shape[2]);
        inner_shift1_type<3>(src, dst, y, x, allow_out, zero_out);
        return dst.self();
      }
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shift() does not support array with \
        " SIZE_T_FMT " dimensions.", info.nd);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(py_shift1_c_overloads, py_shift1_c, 4, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(py_shift1_p_overloads, py_shift1_p, 3, 5)


template <typename T, int N>
static void inner_shift2(bob::python::const_ndarray src,
  bob::python::const_ndarray smask, bob::python::ndarray dst,
  bob::python::ndarray dmask, const int y, const int x, const bool allow_out,
  const bool zero_out)
{
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  bob::ip::shift<T>(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, y, x, 
    allow_out, zero_out);
}

template <int N>
static void inner_shift2_type(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst,
  bob::python::ndarray dmask, const int y, const int x, 
  const bool allow_out, const bool zero_out)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.dtype) {
    case bob::core::array::t_uint8:
      return inner_shift2<uint8_t,N>(src, smask, dst, dmask, y, x, 
                                     allow_out, zero_out);
    case bob::core::array::t_uint16:
      return inner_shift2<uint16_t,N>(src, smask, dst, dmask, y, x,
                                      allow_out, zero_out);
    case bob::core::array::t_float64:
      return inner_shift2<double,N>(src, smask, dst, dmask, y, x,
                                    allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, 
        "bob.ip.shift() does not support array of type '%s'.", 
        info.str().c_str());
  }
}

static void py_shift2_c(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst,
  bob::python::ndarray dmask, const int y, const int x,
  const bool allow_out=false, const bool zero_out=false)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd) {
    case 2: 
      return inner_shift2_type<2>(src, smask, dst, dmask, y, x,
                                 allow_out, zero_out);
    case 3: 
      return inner_shift2_type<3>(src, smask, dst, dmask, y, x,
                                 allow_out, zero_out);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shift() does not support array with \
        " SIZE_T_FMT " dimensions.", info.nd);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(py_shift2_c_overloads, py_shift2_c, 6, 8)


void bind_ip_crop_shift() 
{
  def("crop", &py_crop1_c, 
    py_crop1_c_overloads((arg("src"), arg("dst"), arg("crop_y"), 
      arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false,
      arg("zero_out")=false), 
    "Crop a 2D or 3D array/image. The destination array should have the \
     expected size."));
  def("crop", &py_crop1_p, 
    py_crop1_p_overloads((arg("src"), arg("crop_y"), arg("crop_x"), 
      arg("crop_h"), arg("crop_w"), arg("allow_out")=false, 
      arg("zero_out")=false), 
    "Crop a 2D or 3D array/image. The cropped image will be allocated and \
     returned."));
  def("crop", &py_crop2_c, 
    py_crop2_c_overloads((arg("src"), arg("src_mask"), arg("dst"), 
      arg("dst_mask"), arg("crop_y"), arg("crop_x"), arg("crop_h"), 
      arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false),
    "Crop a 2D or 3D array/image, taking mask into account."));

  def("shift", &py_shift1_c, 
    py_shift1_c_overloads((arg("src"), arg("dst"), arg("shift_y"), 
      arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), 
    "Shift a 2D or 3D array/image. The destination array should have the \
     same size as the source array."));
  def("shift", &py_shift1_p, 
    py_shift1_p_overloads((arg("src"), arg("shift_y"), arg("shift_x"), 
      arg("allow_out")=false, arg("zero_out")=false), 
    "Shift a 2D or 3D array/image. The shifted image will be allocated and \
     returned."));
  def("shift", &py_shift2_c, 
    py_shift2_c_overloads((arg("src"), arg("src_mask"), arg("dst"), 
      arg("dst_mask"), arg("shift_y"), arg("shift_x"), 
      arg("allow_out")=false, arg("zero_out")=false), 
    "Shift a 2 or 3D array/image, taking mask into account."));
}
