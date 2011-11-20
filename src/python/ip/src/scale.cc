/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 20 Nov 18:57:07 2011 CET
 *
 * @brief Binds scaling operation to python 
 */

#include "core/python/ndarray.h"
#include "ip/scale.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

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
  enum_<Torch::ip::Rescale::Algorithm>("RescaleAlgorithm")
    .value("NearesetNeighbour", Torch::ip::Rescale::NearestNeighbour)
    .value("BilinearInterp", Torch::ip::Rescale::BilinearInterp)
    ;

  def("scale", &scale, scale_overloads((arg("src"), arg("dst"), arg("algorithm")="BilinearInterp"), "Rescale a 2D array/image with the given dimensions."));

  def("scale", &scale2, scale2_overloads((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("algorithm")="BilinearInterp"), "Rescale a 2D array/image with the given dimensions, taking mask into account."));

	def("scaleAs", &scale_as, (arg("original"), arg("scale_factor")), "Gives back a scaled version of the original 2 or 3D array (image)");
}
