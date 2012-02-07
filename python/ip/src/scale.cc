/**
 * @file python/ip/src/scale.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds scaling operation to python
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

#include "core/python/ndarray.h"
#include "ip/scale.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ip = bob::ip;
namespace ca = bob::core::array;

template <typename T, int N>
static void inner_scale (tp::const_ndarray src, tp::ndarray dst,
    ip::Rescale::Algorithm algo) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  ip::scale<T>(src.bz<T,N>(), dst_, algo);
}

static void scale (tp::const_ndarray src, tp::ndarray dst,
    ip::Rescale::Algorithm algo=ip::Rescale::BilinearInterp) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "image scaling does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_scale<uint8_t,2>(src, dst, algo);
    case ca::t_uint16:
      return inner_scale<uint16_t,2>(src, dst, algo);
    case ca::t_float64:
      return inner_scale<double,2>(src, dst, algo);
    default:
      PYTHON_ERROR(TypeError, "image scaling does not support type '%s'", info.str().c_str());
  }

}

BOOST_PYTHON_FUNCTION_OVERLOADS(scale_overloads, scale, 2, 3) 

template <typename T, int N>
static void inner_scale2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask, ip::Rescale::Algorithm algo) {
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  ip::scale<T>(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, algo);
}

static void scale2 (tp::const_ndarray src, tp::const_ndarray smask,
    tp::ndarray dst, tp::ndarray dmask,
    ip::Rescale::Algorithm algo=ip::Rescale::BilinearInterp) {

  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "image scaling does not support type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_scale2<uint8_t,2>(src, smask, dst, dmask, algo);
    case ca::t_uint16:
      return inner_scale2<uint16_t,2>(src, smask, dst, dmask, algo);
    case ca::t_float64:
      return inner_scale2<double,2>(src, smask, dst, dmask, algo);
    default:
      PYTHON_ERROR(TypeError, "image scaling does not support type '%s'", info.str().c_str());
  }

}

template <typename T, int N>
static object inner_scale_as (tp::const_ndarray src, double f) {
  return object(ip::scaleAs<T>(src.bz<T,N>(), f)); //copying!
}

template <typename T>
static object inner_scale_as_dim (tp::const_ndarray src, double f) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_scale_as<T,2>(src, f);
    case 3: return inner_scale_as<T,3>(src, f);
    default:
      PYTHON_ERROR(TypeError, "image scaling does not support type '%s'", info.str().c_str());
  }
}

static object scale_as (tp::const_ndarray src, double f) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_scale_as_dim<uint8_t>(src, f);
    case ca::t_uint16:
      return inner_scale_as_dim<uint16_t>(src, f);
    case ca::t_float64:
      return inner_scale_as_dim<double>(src, f);
    default:
      PYTHON_ERROR(TypeError, "image scaling does not support type '%s'", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(scale2_overloads, scale2, 4, 5)

void bind_ip_scale() {
  enum_<bob::ip::Rescale::Algorithm>("RescaleAlgorithm")
    .value("NearesetNeighbour", bob::ip::Rescale::NearestNeighbour)
    .value("BilinearInterp", bob::ip::Rescale::BilinearInterp)
    ;

  def("scale", &scale, scale_overloads((arg("src"), arg("dst"), arg("algorithm")="BilinearInterp"), "Rescale a 2D array/image with the given dimensions."));

  def("scale", &scale2, scale2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("algorithm")="BilinearInterp"), "Rescale a 2D array/image with the given dimensions, taking mask into account."));

	def("scaleAs", &scale_as, (arg("original"), arg("scale_factor")), "Gives back a scaled version of the original 2 or 3D array (image)");
}
