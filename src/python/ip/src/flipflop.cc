/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 20 Nov 17:56:22 2011 CET
 *
 * @brief Binds the flip and flop operations into python 
 */

#include "core/python/ndarray.h"
#include "ip/flipflop.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T, int N>
static void inner_flip (tp::const_ndarray src, tp::ndarray dst) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::flip<T>(src.bz<T,N>(), dst_);
}

template <typename T>
static void inner_flip_dim (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_flip<T,2>(src, dst);
    case 3: return inner_flip<T,3>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flipping does not support type '%s'", info.str().c_str());
  }
}

static void flip (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_flip_dim<uint8_t>(src, dst);
    case ca::t_uint16:
      return inner_flip_dim<uint16_t>(src, dst);
    case ca::t_float64:
      return inner_flip_dim<double>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flipping does not support type '%s'", info.str().c_str());
  }
}

template <typename T, int N>
static void inner_flop (tp::const_ndarray src, tp::ndarray dst) {
  blitz::Array<T,N> dst_ = dst.bz<T,N>();
  ip::flop<T>(src.bz<T,N>(), dst_);
}

template <typename T>
static void inner_flop_dim (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.nd) {
    case 2: return inner_flop<T,2>(src, dst);
    case 3: return inner_flop<T,3>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flopping does not support type '%s'", info.str().c_str());
  }
}

static void flop (tp::const_ndarray src, tp::ndarray dst) {
  const ca::typeinfo& info = src.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_flop_dim<uint8_t>(src, dst);
    case ca::t_uint16:
      return inner_flop_dim<uint16_t>(src, dst);
    case ca::t_float64:
      return inner_flop_dim<double>(src, dst);
    default:
      PYTHON_ERROR(TypeError, "image flopping does not support type '%s'", info.str().c_str());
  }
}

void bind_ip_flipflop() {
  static const char* FLIP_DOC = "Flip a 2 or 3D array/image upside-down.";
  static const char* FLOP_DOC = "Flop a 2 or 3D array/image left-right.";
  def("flip", &flip, (arg("src"), arg("dst")), FLIP_DOC); 
  def("flop", &flop, (arg("src"), arg("dst")), FLOP_DOC); 
}
