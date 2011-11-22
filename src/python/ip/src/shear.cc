/**
 * @file python/ip/src/shear.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds shearing operation into python
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "ip/shear.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T, int N>
static object inner_shear_x_shape (tp::const_ndarray src, double s) {
  return object(ip::getShearXShape<T>(src.bz<T,N>(), s));
}

static object shear_x_shape (tp::const_ndarray src, double s) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_shear_x_shape<uint8_t,2>(src, s);
    case ca::t_uint16: return inner_shear_x_shape<uint16_t,2>(src, s);
    case ca::t_float64: return inner_shear_x_shape<double,2>(src, s);
    default:
      PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static object inner_shear_y_shape (tp::const_ndarray src, double s) {
  return object(ip::getShearYShape<T>(src.bz<T,N>(), s));
}

static object shear_y_shape (tp::const_ndarray src, double s) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_shear_y_shape<uint8_t,2>(src, s);
    case ca::t_uint16: return inner_shear_y_shape<uint16_t,2>(src, s);
    case ca::t_float64: return inner_shear_y_shape<double,2>(src, s);
    default:
      PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_shear_x (tp::const_ndarray src, tp::ndarray dst,
    double a, bool aa) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  ip::shearX<T>(src.bz<T,N>(), dst_, a, aa);
}

static void shear_x (tp::const_ndarray src, tp::ndarray dst, 
    double a, bool aa=true) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2) PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_shear_x<uint8_t,2>(src, dst, a, aa);
    case ca::t_uint16: return inner_shear_x<uint16_t,2>(src, dst, a, aa);
    case ca::t_float64: return inner_shear_x<double,2>(src, dst, a, aa);
    default:
      PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_x_overloads, shear_x, 3, 4) 

template <typename T, int N>
static void inner_shear_y (tp::const_ndarray src, tp::ndarray dst,
    double a, bool aa) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  ip::shearY<T>(src.bz<T,N>(), dst_, a, aa);
}

static void shear_y (tp::const_ndarray src, tp::ndarray dst, 
    double a, bool aa=true) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2) PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_shear_y<uint8_t,2>(src, dst, a, aa);
    case ca::t_uint16: return inner_shear_y<uint16_t,2>(src, dst, a, aa);
    case ca::t_float64: return inner_shear_y<double,2>(src, dst, a, aa);
    default:
      PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_y_overloads, shear_y, 3, 4)

template <typename T, int N>
static void inner_shear_x2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask, double a, bool aa) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  ip::shearX<T>(src.bz<T,N>(), src.bz<bool,N>(), dst_, dmask_, a, aa);
}

static void shear_x2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask, double a, bool aa=true) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2) PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8:
      return inner_shear_x2<uint8_t,2>(src, smask, dst, dmask, a, aa);
    case ca::t_uint16:
      return inner_shear_x2<uint16_t,2>(src, smask, dst, dmask, a, aa);
    case ca::t_float64:
      return inner_shear_x2<double,2>(src, smask, dst, dmask, a, aa);
    default:
      PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_x2_overloads, shear_x2, 5, 6) 

template <typename T, int N>
static void inner_shear_y2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask, double a, bool aa) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  ip::shearY<T>(src.bz<T,N>(), src.bz<bool,N>(), dst_, dmask_, a, aa);
}

static void shear_y2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask, double a, bool aa=true) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2) PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8:
      return inner_shear_y2<uint8_t,2>(src, smask, dst, dmask, a, aa);
    case ca::t_uint16:
      return inner_shear_y2<uint16_t,2>(src, smask, dst, dmask, a, aa);
    case ca::t_float64:
      return inner_shear_y2<double,2>(src, smask, dst, dmask, a, aa);
    default:
      PYTHON_ERROR(TypeError, "shear does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(shear_y2_overloads, shear_x2, 5, 6) 

void bind_ip_shear() {

  def("getShearXShape", &shear_x_shape, (arg("src"), arg("shear")), "Return the shape of the output 2D array/image, when calling shearX.");

  def("getShearYShape", &shear_y_shape, (arg("src"), arg("shear")), "Return the shape of the output 2D array/image, when calling shearY.");

  def("shearX", &shear_x, shear_x_overloads((arg("src"), arg("dst"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the X-dimension."));
  
  def("shearY", &shear_y, shear_y_overloads((arg("src"), arg("dst"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the Y-dimension."));

  def("shearX", &shear_x2, shear_x2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the X-dimension, taking mask into account."));
  
  def("shearX", &shear_y2, shear_y2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("angle"), arg("antialias")=true), "Shear a 2D array/image with the given shear parameter along the Y-dimension, taking mask into account."));

}
