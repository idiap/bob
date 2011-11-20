/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 20 Nov 18:38:41 2011 CET
 *
 * @brief Binds integral image implementation to python 
 */

#include "core/python/ndarray.h"
#include "ip/integral.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T, typename U, int N>
static void inner_integral (tp::const_ndarray src, tp::ndarray dst, bool b) {
  blitz::Array<U,N> dst_ = dst.bz<U,N>();
  ip::integral(src.bz<T,N>(), dst_, b);
}

template <typename T, int N>
static void integral2 (tp::const_ndarray src, tp::ndarray dst, bool b) {
  const ca::typeinfo& info = dst.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "integral image operator does not support output type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_int8: return inner_integral<T,int8_t,N>(src, dst, b);
    case ca::t_int16: return inner_integral<T,int16_t,N>(src, dst, b);
    case ca::t_int32: return inner_integral<T,int32_t,N>(src, dst, b);
    case ca::t_int64: return inner_integral<T,int64_t,N>(src, dst, b);
    case ca::t_uint8: return inner_integral<T,uint8_t,N>(src, dst, b);
    case ca::t_uint16: return inner_integral<T,uint16_t,N>(src, dst, b);
    case ca::t_uint32: return inner_integral<T,uint32_t,N>(src, dst, b);
    case ca::t_uint64: return inner_integral<T,uint64_t,N>(src, dst, b);
    case ca::t_float32: return inner_integral<T,float,N>(src, dst, b);
    case ca::t_float64: return inner_integral<T,double,N>(src, dst, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support output type '%s'", info.str().c_str());
  }

}

static void integral (tp::const_ndarray src, tp::ndarray dst, bool b=false) {
  const ca::typeinfo& info = src.type();

  if (info.nd != 2)
    PYTHON_ERROR(TypeError, "integral image operator does not support input type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return integral2<uint8_t,2>(src, dst, b);
    case ca::t_uint16: return integral2<uint16_t,2>(src, dst, b);
    case ca::t_float64: return integral2<double,2>(src, dst, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support input type '%s'", info.str().c_str());
  }

}

BOOST_PYTHON_FUNCTION_OVERLOADS(integral_overloads, integral, 2, 3)

void bind_ip_integral() {
  def(BOOST_PP_STRINGIZE(integral), &integral, integral_overloads((arg("src"), arg("dst"), arg("addZeroBorder")=false), "Compute the integral image of a 2D blitz array (image).")); 
}
