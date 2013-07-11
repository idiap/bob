/**
 * @file ip/python/shear.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds shearing operation into python
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

#include <bob/python/ndarray.h>
#include <bob/ip/shear.h>

using namespace boost::python;

template <typename T, int N>
static object inner_shear_x_shape(bob::python::const_ndarray src, double s) 
{
  return object(bob::ip::getShearXShape<T>(src.bz<T,N>(), s));
}

static object shear_x_shape(bob::python::const_ndarray src, double s) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8: return inner_shear_x_shape<uint8_t,2>(src, s);
    case bob::core::array::t_uint16: return inner_shear_x_shape<uint16_t,2>(src, s);
    case bob::core::array::t_float64: return inner_shear_x_shape<double,2>(src, s);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.get_shear_x_shape() does not support array of type '%s'.", info.str().c_str());
  }
}

template <typename T, int N>
static object inner_shear_y_shape (bob::python::const_ndarray src, double s) 
{
  return object(bob::ip::getShearYShape<T>(src.bz<T,N>(), s));
}

static object shear_y_shape (bob::python::const_ndarray src, double s) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8: return inner_shear_y_shape<uint8_t,2>(src, s);
    case bob::core::array::t_uint16: return inner_shear_y_shape<uint16_t,2>(src, s);
    case bob::core::array::t_float64: return inner_shear_y_shape<double,2>(src, s);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.get_shear_y_shape() does not support array of type '%s'.", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_shear_x(bob::python::const_ndarray src, 
  bob::python::ndarray dst, double a, bool aa) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  bob::ip::shearX<T>(src.bz<T,N>(), dst_, a, aa);
}

static void shear_x(bob::python::const_ndarray src, 
  bob::python::ndarray dst, double a, bool aa=true) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8: return inner_shear_x<uint8_t,2>(src, dst, a, aa);
    case bob::core::array::t_uint16: return inner_shear_x<uint16_t,2>(src, dst, a, aa);
    case bob::core::array::t_float64: return inner_shear_x<double,2>(src, dst, a, aa);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shear_x() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_x_overloads, shear_x, 3, 4) 

template <typename T, int N>
static object inner_shear_x_p(bob::python::const_ndarray src, double a, 
  bool aa)
{
  const blitz::TinyVector<int,2> shape = bob::ip::getShearXShape<T>(src.bz<T,2>(), a);
  bob::python::ndarray dst(bob::core::array::t_float64, shape(0), shape(1));
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  bob::ip::shearX<T>(src.bz<T,N>(), dst_, a, aa);
  return dst.self();
}

static object shear_x_p(bob::python::const_ndarray src, double a, 
  bool aa=true)
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8: return inner_shear_x_p<uint8_t,2>(src, a, aa);
    case bob::core::array::t_uint16: return inner_shear_x_p<uint16_t,2>(src, a, aa);
    case bob::core::array::t_float64: return inner_shear_x_p<double,2>(src, a, aa);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shear_x() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_x_p_overloads, shear_x_p, 2, 3) 


template <typename T, int N>
static void inner_shear_y(bob::python::const_ndarray src, 
  bob::python::ndarray dst, double a, bool aa) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  bob::ip::shearY<T>(src.bz<T,N>(), dst_, a, aa);
}

static void shear_y(bob::python::const_ndarray src, 
  bob::python::ndarray dst, double a, bool aa=true) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8: return inner_shear_y<uint8_t,2>(src, dst, a, aa);
    case bob::core::array::t_uint16: return inner_shear_y<uint16_t,2>(src, dst, a, aa);
    case bob::core::array::t_float64: return inner_shear_y<double,2>(src, dst, a, aa);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shear_y() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_y_overloads, shear_y, 3, 4)

template <typename T, int N>
static object inner_shear_y_p(bob::python::const_ndarray src, double a, 
  bool aa)
{
  const blitz::TinyVector<int,2> shape = bob::ip::getShearYShape<T>(src.bz<T,2>(), a);
  bob::python::ndarray dst(bob::core::array::t_float64, shape(0), shape(1));
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  bob::ip::shearY<T>(src.bz<T,N>(), dst_, a, aa);
  return dst.self();
}

static object shear_y_p(bob::python::const_ndarray src, double a, 
  bool aa=true)
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8: return inner_shear_y_p<uint8_t,2>(src, a, aa);
    case bob::core::array::t_uint16: return inner_shear_y_p<uint16_t,2>(src, a, aa);
    case bob::core::array::t_float64: return inner_shear_y_p<double,2>(src, a, aa);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shear_y() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_y_p_overloads, shear_y_p, 2, 3) 

template <typename T, int N>
static void inner_shear_x2 (bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst,
  bob::python::ndarray dmask, double a, bool aa) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  bob::ip::shearX<T>(src.bz<T,N>(), src.bz<bool,N>(), dst_, dmask_, a, aa);
}

static void shear_x2(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst, 
  bob::python::ndarray dmask, double a, bool aa=true) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8:
      return inner_shear_x2<uint8_t,2>(src, smask, dst, dmask, a, aa);
    case bob::core::array::t_uint16:
      return inner_shear_x2<uint16_t,2>(src, smask, dst, dmask, a, aa);
    case bob::core::array::t_float64:
      return inner_shear_x2<double,2>(src, smask, dst, dmask, a, aa);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shear_x() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_x2_overloads, shear_x2, 5, 6) 

template <typename T, int N>
static void inner_shear_y2(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst, 
  bob::python::ndarray dmask, double a, bool aa) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  bob::ip::shearY<T>(src.bz<T,N>(), src.bz<bool,N>(), dst_, dmask_, a, aa);
}

static void shear_y2(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, bob::python::ndarray dst, 
  bob::python::ndarray dmask, double a, bool aa=true) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.dtype) {
    case bob::core::array::t_uint8:
      return inner_shear_y2<uint8_t,2>(src, smask, dst, dmask, a, aa);
    case bob::core::array::t_uint16:
      return inner_shear_y2<uint16_t,2>(src, smask, dst, dmask, a, aa);
    case bob::core::array::t_float64:
      return inner_shear_y2<double,2>(src, smask, dst, dmask, a, aa);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.shear_y() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_y2_overloads, shear_x2, 5, 6) 

void bind_ip_shear() 
{
  def("get_shear_x_shape", &shear_x_shape, (arg("src"), arg("shear")), "Returns the shape of the output 2D array/image, when calling shear_x.");
  def("get_shear_y_shape", &shear_y_shape, (arg("src"), arg("shear")), "Returns the shape of the output 2D array/image, when calling shear_y.");
  def("shear_x", &shear_x, shear_x_overloads((arg("src"), arg("dst"), arg("angle"), arg("antialias")=true), "Shears a 2D array/image with the given shear parameter along the X-dimension. The dst array should have the expected size (given by get_shear_x_shape)."));
  def("shear_y", &shear_y, shear_y_overloads((arg("src"), arg("dst"), arg("angle"), arg("antialias")=true), "Shears a 2D array/image with the given shear parameter along the Y-dimension. The dst array should have the expected size (given by get_shear_y_shape)."));
  def("shear_x", &shear_x_p, shear_x_p_overloads((arg("src"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the X-dimension. The destination array is allocated and returned."));
  def("shear_y", &shear_y_p, shear_y_p_overloads((arg("src"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the Y-dimension. The destination array is allocated and returned."));
  def("shear_x", &shear_x2, shear_x2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the X-dimension, taking mask into account."));
  def("shear_y", &shear_y2, shear_y2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the Y-dimension, taking mask into account."));
}
