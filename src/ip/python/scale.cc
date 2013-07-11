/**
 * @file ip/python/scale.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds scaling operation to python
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
#include <bob/ip/scale.h>

using namespace boost::python;

template <typename T, int N>
static void inner_scale(bob::python::const_ndarray src, 
  bob::python::ndarray dst, bob::ip::Rescale::Algorithm algo)
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  bob::ip::scale(src.bz<T,N>(), dst_, algo);
}

static void scale(bob::python::const_ndarray src, bob::python::ndarray dst,
  bob::ip::Rescale::Algorithm algo=bob::ip::Rescale::BilinearInterp) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.nd)
  {
    case 2:
      switch(info.dtype) 
      {
        case bob::core::array::t_uint8: 
          return inner_scale<uint8_t,2>(src, dst, algo);
        case bob::core::array::t_uint16:
          return inner_scale<uint16_t,2>(src, dst, algo);
        case bob::core::array::t_float64:
          return inner_scale<double,2>(src, dst, algo);
        default:
          PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array with type '%s'.", info.str().c_str());
      }
      break;
    case 3:
      switch(info.dtype) 
      {
        case bob::core::array::t_uint8: 
          return inner_scale<uint8_t,3>(src, dst, algo);
        case bob::core::array::t_uint16:
          return inner_scale<uint16_t,3>(src, dst, algo);
        case bob::core::array::t_float64:
          return inner_scale<double,3>(src, dst, algo);
        default:
          PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array with type '%s'.", info.str().c_str());
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array with " SIZE_T_FMT " dimensions.", info.nd);
  }

}

BOOST_PYTHON_FUNCTION_OVERLOADS(scale_overloads, scale, 2, 3) 

template <typename T, int N>
static void inner_scale2(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask, 
  bob::python::ndarray dst, bob::python::ndarray dmask, 
  bob::ip::Rescale::Algorithm algo) 
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  blitz::Array<bool,N> dmask_ = dmask.bz<bool,N>();
  bob::ip::scale(src.bz<T,N>(), smask.bz<bool,N>(), dst_, dmask_, algo);
}

static void scale2(bob::python::const_ndarray src, 
  bob::python::const_ndarray smask,
  bob::python::ndarray dst, bob::python::ndarray dmask,
  bob::ip::Rescale::Algorithm algo=bob::ip::Rescale::BilinearInterp) 
{
  const bob::core::array::typeinfo& info = src.type();

  switch(info.nd)
  {
    case 2:
      switch(info.dtype) 
      {
        case bob::core::array::t_uint8: 
          return inner_scale2<uint8_t,2>(src, smask, dst, dmask, algo);
        case bob::core::array::t_uint16:
          return inner_scale2<uint16_t,2>(src, smask, dst, dmask, algo);
        case bob::core::array::t_float64:
          return inner_scale2<double,2>(src, smask, dst, dmask, algo);
        default:
          PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array of type '%s'.", info.str().c_str());
      }
      break;
    case 3:
      switch(info.dtype) 
      {
        case bob::core::array::t_uint8: 
          return inner_scale2<uint8_t,3>(src, smask, dst, dmask, algo);
        case bob::core::array::t_uint16:
          return inner_scale2<uint16_t,3>(src, smask, dst, dmask, algo);
        case bob::core::array::t_float64:
          return inner_scale2<double,3>(src, smask, dst, dmask, algo);
        default:
          PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array of type '%s'.", info.str().c_str());
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array with " SIZE_T_FMT " dimensions.", info.nd);
  }
}

template <typename T, int N>
static object inner_scale_as(bob::python::const_ndarray src, const double f) 
{
  return object(bob::ip::scaleAs(src.bz<T,N>(), f)); //copying!
}

template <typename T>
static object inner_scale_as_dim(bob::python::const_ndarray src, const double f)
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.nd) 
  {
    case 2: return inner_scale_as<T,2>(src, f);
    case 3: return inner_scale_as<T,3>(src, f);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array with " SIZE_T_FMT " dimensions.", info.nd);
  }
}

static object scale_as(bob::python::const_ndarray src, double f) {
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) 
  {
    case bob::core::array::t_uint8: 
      return inner_scale_as_dim<uint8_t>(src, f);
    case bob::core::array::t_uint16:
      return inner_scale_as_dim<uint16_t>(src, f);
    case bob::core::array::t_float64:
      return inner_scale_as_dim<double>(src, f);
    default:
      PYTHON_ERROR(TypeError, "bob.ip.scale() does not support array of type '%s'.", info.str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(scale2_overloads, scale2, 4, 5)

void bind_ip_scale() 
{
  enum_<bob::ip::Rescale::Algorithm>("RescaleAlgorithm")
    .value("NearesetNeighbour", bob::ip::Rescale::NearestNeighbour)
    .value("BilinearInterp", bob::ip::Rescale::BilinearInterp)
    ;

  def("scale", &scale, scale_overloads((arg("src"), arg("dst"), arg("algorithm")=bob::ip::Rescale::BilinearInterp), "Scales a 2D array/image to the dimensions given by the destination array."));
  def("scale", &scale2, scale2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("algorithm")=bob::ip::Rescale::BilinearInterp), "Scales a 2D array/image to the dimensions given by the destination array, taking boolean mask into account."));
  def("scale_as", &scale_as, (arg("original"), arg("scale_factor")), "Gives back a scaled version of the original 2D or 3D array/image.");
}
