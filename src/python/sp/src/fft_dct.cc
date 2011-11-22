/**
 * @file python/sp/src/fft_dct.cc
 * @date Thu Apr 14 13:39:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Fast Fourier Transform and the (fast) Discrete Cosine
 * Transform to python.
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

#include <boost/python.hpp>

#include "sp/FFT1D.h"
#include "sp/FFT2D.h"
#include "sp/DCT1D.h"
#include "sp/DCT2D.h"
#include "sp/FFT1DNaive.h"
#include "sp/FFT2DNaive.h"
#include "sp/DCT1DNaive.h"
#include "sp/DCT2DNaive.h"
#include "sp/fftshift.h"

#include "core/python/ndarray.h"

using namespace boost::python;
namespace sp = Torch::sp;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

// documentation for classes
static const char* FFT1D_DOC = "Objects of this class, after configuration, can compute the direct FFT of a 1D array/signal.";
static const char* IFFT1D_DOC = "Objects of this class, after configuration, can compute the inverse FFT of a 1D array/signal.";
static const char* FFT2D_DOC = "Objects of this class, after configuration, can compute the direct FFT of a 2D array/signal.";
static const char* IFFT2D_DOC = "Objects of this class, after configuration, can compute the inverse FFT of a 2D array/signal.";
 
static const char* DCT1D_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 1D array/signal.";
static const char* IDCT1D_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 1D array/signal.";
static const char* DCT2D_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 2D array/signal.";
static const char* IDCT2D_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 2D array/signal.";

// free methods documentation
static const char* FFT_DOC = "Compute the direct FFT of a 1 or 2D array/signal of type complex128.";
static const char* IFFT_DOC = "Compute the inverse FFT of a 1 or 2D array/signalof type complex128.";

static const char* DCT_DOC = "Compute the direct DCT of a 1 or 2D array/signal of type float64.";
static const char* IDCT_DOC = "Compute the inverse DCT of a 1 or 2D array/signal of type float64.";

static const char* FFTSHIFT_DOC = "If a 1D complex128 array is passed, inverses the two halves of that array and returns the result as a new array. If a 2D complex128 array is passed, swaps the four quadrants of the array and returns the result as a new array.";
static const char* IFFTSHIFT_DOC = "This method undo what fftshift() does. Accepts 1 or 2D array of type complex128.";

static object script_fft(tp::const_ndarray ar) {
  typedef std::complex<double> dcplx;
  const ca::typeinfo& info = ar.type();
  if (info.dtype != ca::t_complex128) {
    PYTHON_ERROR(TypeError, "FFT operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  tp::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        Torch::sp::FFT1D op(info.shape[0]);
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        op(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        Torch::sp::FFT2D op(info.shape[0], info.shape[1]);
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        op(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "FFT operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  return res.self();
}

static object script_ifft(tp::const_ndarray ar) {
  typedef std::complex<double> dcplx;
  const ca::typeinfo& info = ar.type();
  if (info.dtype != ca::t_complex128) {
    PYTHON_ERROR(TypeError, "iFFT operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  tp::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        Torch::sp::IFFT1D op(info.shape[0]);
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        op(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        Torch::sp::IFFT2D op(info.shape[0], info.shape[1]);
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        op(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iFFT operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  return res.self();
}

static object script_dct(tp::const_ndarray ar) {
  const ca::typeinfo& info = ar.type();
  if (info.dtype != ca::t_float64) {
    PYTHON_ERROR(TypeError, "DCT operation only supports 1 or 2D double input arrays - you provided '%s'", info.str().c_str());
  }
  tp::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        Torch::sp::DCT1D op(info.shape[0]);
        blitz::Array<double,1> res_ = res.bz<double,1>();
        op(ar.bz<double,1>(), res_);
      }
      break;
    case 2:
      {
        Torch::sp::DCT2D op(info.shape[0], info.shape[1]);
        blitz::Array<double,2> res_ = res.bz<double,2>();
        op(ar.bz<double,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "DCT operation only supports 1 or 2D double input arrays - you provided '%s'", info.str().c_str());
  }
  return res.self();
}

static object script_idct(tp::const_ndarray ar) {
  const ca::typeinfo& info = ar.type();
  if (info.dtype != ca::t_float64) {
    PYTHON_ERROR(TypeError, "iDCT operation only supports 1 or 2D double input arrays - you provided '%s'", info.str().c_str());
  }
  tp::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        Torch::sp::IDCT1D op(info.shape[0]);
        blitz::Array<double,1> res_ = res.bz<double,1>();
        op(ar.bz<double,1>(), res_);
      }
      break;
    case 2:
      {
        Torch::sp::IDCT2D op(info.shape[0], info.shape[1]);
        blitz::Array<double,2> res_ = res.bz<double,2>();
        op(ar.bz<double,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iDCT operation only supports 1 or 2D double input arrays - you provided '%s'", info.str().c_str());
  }
  return res.self();
}

static object script_fftshift(tp::const_ndarray ar) {
  typedef std::complex<double> dcplx;
  const ca::typeinfo& info = ar.type();
  if (info.dtype != ca::t_complex128) {
    PYTHON_ERROR(TypeError, "FFTshift operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  tp::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        sp::fftshift(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        sp::fftshift(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "FFTshift operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  return res.self();
}

static object script_ifftshift(tp::const_ndarray ar) {
  typedef std::complex<double> dcplx;
  const ca::typeinfo& info = ar.type();
  if (info.dtype != ca::t_complex128) {
    PYTHON_ERROR(TypeError, "iFFTshift operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  tp::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        sp::ifftshift(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        sp::ifftshift(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iFFTshift operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
  return res.self();
}

static void py_fft1d_call(sp::FFT1D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<std::complex<double>,1> c_ = c.bz<std::complex<double>,1>();
  a(b.bz<std::complex<double>,1>(), c_);
}

static void py_ifft1d_call(sp::IFFT1D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<std::complex<double>,1> c_ = c.bz<std::complex<double>,1>();
  a(b.bz<std::complex<double>,1>(), c_);
}

static void py_fft2d_call(sp::FFT2D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<std::complex<double>,2> c_ = c.bz<std::complex<double>,2>();
  a(b.bz<std::complex<double>,2>(), c_);
}

static void py_ifft2d_call(sp::IFFT2D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<std::complex<double>,2> c_ = c.bz<std::complex<double>,2>();
  a(b.bz<std::complex<double>,2>(), c_);
}

static void py_dct1d_call(sp::DCT1D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<double,1> c_ = c.bz<double,1>();
  a(b.bz<double,1>(), c_);
}

static void py_idct1d_call(sp::IDCT1D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<double,1> c_ = c.bz<double,1>();
  a(b.bz<double,1>(), c_);
}

static void py_dct2d_call(sp::DCT2D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<double,2> c_ = c.bz<double,2>();
  a(b.bz<double,2>(), c_);
}

static void py_idct2d_call(sp::IDCT2D& a, tp::const_ndarray b, tp::ndarray c) {
  blitz::Array<double,2> c_ = c.bz<double,2>();
  a(b.bz<double,2>(), c_);
}

static void py_fftshift(tp::const_ndarray ar, tp::ndarray t) {
  const ca::typeinfo& info = ar.type();
  switch (info.nd) {
    case 1:
      {
        blitz::Array<std::complex<double>,1> t_ =
          t.bz<std::complex<double>,1>();
        sp::fftshift(ar.bz<std::complex<double>,1>(), t_);
      }
      break;
    case 2:
      {
        blitz::Array<std::complex<double>,2> t_ =
          t.bz<std::complex<double>,2>();
        sp::fftshift(ar.bz<std::complex<double>,2>(), t_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "FFTshift operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
}

static void py_ifftshift(tp::const_ndarray ar, tp::ndarray t) {
  const ca::typeinfo& info = ar.type();
  switch (info.nd) {
    case 1:
      {
        blitz::Array<std::complex<double>,1> t_ =
          t.bz<std::complex<double>,1>();
        sp::ifftshift(ar.bz<std::complex<double>,1>(), t_);
      }
      break;
    case 2:
      {
        blitz::Array<std::complex<double>,2> t_ =
          t.bz<std::complex<double>,2>();
        sp::ifftshift(ar.bz<std::complex<double>,2>(), t_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iFFTshift operation only supports 1 or 2D complex128 input arrays - you provided '%s'", info.str().c_str());
  }
}

void bind_sp_fft_dct()
{
  // Fast Fourier Transform
  class_<sp::FFT1DAbstract, boost::noncopyable>("FFT1DAbstract", "Abstract class for FFT1D", no_init)
    .def("reset", (void (sp::FFT1D::*)(const int))&sp::FFT1D::reset, (arg("self"),arg("length")), "Reset the length of the expected input signals.")
    .add_property("length", &sp::FFT1D::getLength)
    ;

  class_<sp::FFT1D, boost::shared_ptr<sp::FFT1D>, bases<sp::FFT1DAbstract> >("FFT1D", FFT1D_DOC, init<const int>((arg("length"))))
    .def("__call__", &py_fft1d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the FFT of the input 1D array/signal.")
    ;

  class_<sp::IFFT1D, boost::shared_ptr<sp::IFFT1D>, bases<sp::FFT1DAbstract> >("IFFT1D", IFFT1D_DOC, init<const int>((arg("length"))))
    .def("__call__", &py_ifft1d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the IFFT of the input 1D array/signal.")
    ;

  class_<sp::FFT2DAbstract, boost::noncopyable>("FFT2DAbstract", "Abstract class for FFT2D", no_init)
    .def("reset", (void (sp::FFT2D::*)(const int, const int))&sp::FFT2D::reset, (arg("self"), arg("height"), arg("width")), "Reset the dimension of the expected input signals.")
    .add_property("height", &sp::FFT2D::getHeight)
    .add_property("width", &sp::FFT2D::getWidth)
    ;

  class_<sp::FFT2D, boost::shared_ptr<sp::FFT2D>, bases<sp::FFT2DAbstract> >("FFT2D", FFT2D_DOC, init<const int,const int>((arg("height"), arg("width"))))
    .def("__call__", &py_fft2d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the FFT of the input 1D array/signal.")
    ;

  class_<sp::IFFT2D, boost::shared_ptr<sp::IFFT2D>, bases<sp::FFT2DAbstract> >("IFFT2D", IFFT2D_DOC, init<const int,const int>((arg("height"), arg("width"))))
    .def("__call__", &py_ifft2d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the IFFT of the input 1D array/signal.")
    ;

  // (Fast) Discrete Cosine Transform
  class_<sp::DCT1DAbstract, boost::noncopyable>("DCT1DAbstract", "Abstract class for DCT1D", no_init)
    .def("reset", (void (sp::DCT1D::*)(const int))&sp::DCT1D::reset, (arg("self"),arg("length")), "Reset the length of the expected input signals.")
    .add_property("length", &sp::DCT1D::getLength)
    ;

  class_<sp::DCT1D, boost::shared_ptr<sp::DCT1D>, bases<sp::DCT1DAbstract> >("DCT1D", DCT1D_DOC, init<const int>((arg("length"))))
    .def("__call__", &py_dct1d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the DCT of the input 1D array/signal.")
    ;

  class_<sp::IDCT1D, boost::shared_ptr<sp::IDCT1D>, bases<sp::DCT1DAbstract> >("IDCT1D", IDCT1D_DOC, init<const int>((arg("length"))))
    .def("__call__", &py_idct1d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the IDCT of the input 1D array/signal.")
    ;

  class_<sp::DCT2DAbstract, boost::noncopyable>("DCT2DAbstract", "Abstract class for DCT2D", no_init)
    .def("reset", (void (sp::DCT2D::*)(const int, const int))&sp::DCT2D::reset, (arg("self"), arg("height"), arg("width")), "Reset the dimension of the expected input signals.")
    .add_property("height", &sp::DCT2D::getHeight)
    .add_property("width", &sp::DCT2D::getWidth)
    ;

  class_<sp::DCT2D, boost::shared_ptr<sp::DCT2D>, bases<sp::DCT2DAbstract> >("DCT2D", DCT2D_DOC, init<const int, const int>((arg("height"), arg("width"))))
    .def("__call__", &py_dct2d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the DCT of the input 1D array/signal.")
    ;

  class_<sp::IDCT2D, boost::shared_ptr<sp::IDCT2D>, bases<sp::DCT2DAbstract> >("IDCT2D", IDCT2D_DOC, init<const int, const int>((arg("height"), arg("width"))))
    .def("__call__", &py_idct2d_call, (arg("self"),arg("input"), arg("output")), "Call an object of this type to compute the IDCT of the input 1D array/signal.")
    ;

  // fft and dct function-like 
  def("fft", &script_fft, (arg("array")), FFT_DOC);
  def("ifft", &script_ifft, (arg("array")), IFFT_DOC);

  def("dct", &script_dct, (arg("array")), DCT_DOC);
  def("idct", &script_idct, (arg("array")), IDCT_DOC);


  // fftshift
  def("fftshift", &script_fftshift, (arg("array")), FFTSHIFT_DOC);
  def("ifftshift", &script_ifftshift, (arg("array")), IFFTSHIFT_DOC);

  def("fftshift", &py_fftshift, (arg("input"),arg("output")), FFTSHIFT_DOC);
  def("ifftshift", &py_ifftshift, (arg("input"),arg("output")), IFFTSHIFT_DOC);
}
