/**
 * @file ip/python/integral.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds integral image implementation to python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/python/ndarray.h>
#include <bob/ip/integral.h>

using namespace boost::python;

template <typename T, typename U, int N>
static void inner_integral (bob::python::const_ndarray src, bob::python::ndarray dst, bool b) {
  blitz::Array<U,N> dst_ = dst.bz<U,N>();
  bob::ip::integral(src.bz<T,N>(), dst_, b);
}

template <typename T, int N>
static void integral2 (bob::python::const_ndarray src, bob::python::ndarray dst, bool b) {
  const bob::core::array::typeinfo& info = dst.type();

  switch (info.dtype) {
    case bob::core::array::t_int8: return inner_integral<T,int8_t,N>(src, dst, b);
    case bob::core::array::t_int16: return inner_integral<T,int16_t,N>(src, dst, b);
    case bob::core::array::t_int32: return inner_integral<T,int32_t,N>(src, dst, b);
    case bob::core::array::t_int64: return inner_integral<T,int64_t,N>(src, dst, b);
    case bob::core::array::t_uint8: return inner_integral<T,uint8_t,N>(src, dst, b);
    case bob::core::array::t_uint16: return inner_integral<T,uint16_t,N>(src, dst, b);
    case bob::core::array::t_uint32: return inner_integral<T,uint32_t,N>(src, dst, b);
    case bob::core::array::t_uint64: return inner_integral<T,uint64_t,N>(src, dst, b);
    case bob::core::array::t_float32: return inner_integral<T,float,N>(src, dst, b);
    case bob::core::array::t_float64: return inner_integral<T,double,N>(src, dst, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support output type '%s'", info.str().c_str());
  }

}

static void integral (bob::python::const_ndarray src, bob::python::ndarray dst, bool b) {
  const bob::core::array::typeinfo& info = src.type();

  switch (info.dtype) {
    case bob::core::array::t_uint8: return integral2<uint8_t,2>(src, dst, b);
    case bob::core::array::t_uint16: return integral2<uint16_t,2>(src, dst, b);
    case bob::core::array::t_float64: return integral2<double,2>(src, dst, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support input type '%s'", info.str().c_str());
  }

}


template <typename T, typename U, int N>
static void inner_integral_square (bob::python::const_ndarray src, bob::python::ndarray dst, bob::python::ndarray sqr, bool b) {
  blitz::Array<U,N> dst_ = dst.bz<U,N>();
  bob::ip::integral(src.bz<T,N>(), dst_, b);
}

template <typename T, int N>
static void integral2_square (bob::python::const_ndarray src, bob::python::ndarray dst, bob::python::ndarray sqr, bool b) {
  const bob::core::array::typeinfo& info = dst.type();

  switch (info.dtype) {
    case bob::core::array::t_int8: return inner_integral_square<T,int8_t,N>(src, dst, sqr, b);
    case bob::core::array::t_int16: return inner_integral_square<T,int16_t,N>(src, dst, sqr, b);
    case bob::core::array::t_int32: return inner_integral_square<T,int32_t,N>(src, dst, sqr, b);
    case bob::core::array::t_int64: return inner_integral_square<T,int64_t,N>(src, dst, sqr, b);
    case bob::core::array::t_uint8: return inner_integral_square<T,uint8_t,N>(src, dst, sqr, b);
    case bob::core::array::t_uint16: return inner_integral_square<T,uint16_t,N>(src, dst, sqr, b);
    case bob::core::array::t_uint32: return inner_integral_square<T,uint32_t,N>(src, dst, sqr, b);
    case bob::core::array::t_uint64: return inner_integral_square<T,uint64_t,N>(src, dst, sqr, b);
    case bob::core::array::t_float32: return inner_integral_square<T,float,N>(src, dst, sqr, b);
    case bob::core::array::t_float64: return inner_integral_square<T,double,N>(src, dst, sqr, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support output type '%s'", info.str().c_str());
  }

}

static void integral_square (bob::python::const_ndarray src, bob::python::ndarray dst, bob::python::ndarray sqr, bool b) {
  const bob::core::array::typeinfo& info = src.type();

  switch (info.dtype) {
    case bob::core::array::t_uint8: return integral2_square<uint8_t,2>(src, dst, sqr, b);
    case bob::core::array::t_uint16: return integral2_square<uint16_t,2>(src, dst, sqr, b);
    case bob::core::array::t_float64: return integral2_square<double,2>(src, dst, sqr, b);
    default:
      PYTHON_ERROR(TypeError, "integral image operator does not support input type '%s'", info.str().c_str());
  }

}


void bind_ip_integral() {
  def("integral", &integral, (arg("src"), arg("dst"), arg("add_zero_border")=false), "Compute the integral image of a 2D blitz array (image). It is the responsibility of the user to select an appropriate type for the numpy array which will contain the integral image. By default, src and dst should have the same size. If add_zero_border is set to true, then dst should be one pixel larger than src in each dimension.");
  def("integral", &integral_square, (arg("src"), arg("dst"), arg("sqr"), arg("add_zero_border")=false), "Compute the integral image and the integral square image of a 2D blitz array (image). It is the responsibility of the user to select an appropriate type for the numpy array which will contain the integral image and the integral square image. By default, src, dst and sqr should have the same size. If add_zero_border is set to true, then dst and sqr should be one pixel larger than src in each dimension.");
}
