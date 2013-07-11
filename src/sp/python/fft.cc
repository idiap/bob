/**
 * @file sp/python/fft.cc
 * @date Thu Apr 14 13:39:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Fast Fourier Transform and the (fast) Discrete Cosine
 * Transform to python.
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

#include <bob/sp/FFT1D.h>
#include <bob/sp/FFT2D.h>
#include <bob/sp/FFT1DNaive.h>
#include <bob/sp/FFT2DNaive.h>
#include <bob/sp/fftshift.h>


using namespace boost::python;

// documentation for classes
static const char* FFT1D_DOC = "Objects of this class, after configuration, can compute the direct FFT of a 1D array/signal.";
static const char* IFFT1D_DOC = "Objects of this class, after configuration, can compute the inverse FFT of a 1D array/signal.";
static const char* FFT2D_DOC = "Objects of this class, after configuration, can compute the direct FFT of a 2D array/signal.";
static const char* IFFT2D_DOC = "Objects of this class, after configuration, can compute the inverse FFT of a 2D array/signal.";
 
// free methods documentation
static const char* FFT_DOC = "Compute the direct FFT of a 1 or 2D array/signal of type complex128.";
static const char* IFFT_DOC = "Compute the inverse FFT of a 1 or 2D array/signalof type complex128.";

static const char* FFTSHIFT_DOC = "If a 1D complex128 array is passed, inverses the two halves of that array and returns the result as a new array. If a 2D complex128 array is passed, swaps the four quadrants of the array and returns the result as a new array.";
static const char* IFFTSHIFT_DOC = "This method undo what fftshift() does. Accepts 1 or 2D array of type complex128.";


static void py_fft1d_c(bob::sp::FFT1D& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<std::complex<double>,1> dst_ = dst.bz<std::complex<double>,1>();
  op(src.bz<std::complex<double>,1>(), dst_);
}

static object py_fft1d_p(bob::sp::FFT1D& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_complex128, op.getLength());
  blitz::Array<std::complex<double>,1> dst_ = dst.bz<std::complex<double>,1>();
  op(src.bz<std::complex<double>,1>(), dst_);
  return dst.self();
}

static void py_ifft1d_c(bob::sp::IFFT1D& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<std::complex<double>,1> dst_ = dst.bz<std::complex<double>,1>();
  op(src.bz<std::complex<double>,1>(), dst_);
}

static object py_ifft1d_p(bob::sp::IFFT1D& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_complex128, op.getLength());
  blitz::Array<std::complex<double>,1> dst_ = dst.bz<std::complex<double>,1>();
  op(src.bz<std::complex<double>,1>(), dst_);
  return dst.self();
}


static void py_fft2d_c(bob::sp::FFT2D& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<std::complex<double>,2> dst_ = dst.bz<std::complex<double>,2>();
  op(src.bz<std::complex<double>,2>(), dst_);
}

static object py_fft2d_p(bob::sp::FFT2D& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_complex128, op.getHeight(), 
    op.getWidth());
  blitz::Array<std::complex<double>,2> dst_ = dst.bz<std::complex<double>,2>();
  op(src.bz<std::complex<double>,2>(), dst_);
  return dst.self();
}

static void py_ifft2d_c(bob::sp::IFFT2D& op, bob::python::const_ndarray src,
  bob::python::ndarray dst) 
{
  blitz::Array<std::complex<double>,2> dst_ = dst.bz<std::complex<double>,2>();
  op(src.bz<std::complex<double>,2>(), dst_);
}

static object py_ifft2d_p(bob::sp::IFFT2D& op, bob::python::const_ndarray src)
{
  bob::python::ndarray dst(bob::core::array::t_complex128, op.getHeight(), 
    op.getWidth());
  blitz::Array<std::complex<double>,2> dst_ = dst.bz<std::complex<double>,2>();
  op(src.bz<std::complex<double>,2>(), dst_);
  return dst.self();
}


static object script_fft(bob::python::const_ndarray ar) 
{
  typedef std::complex<double> dcplx;
  const bob::core::array::typeinfo& info = ar.type();
  bob::python::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        bob::sp::FFT1D op(info.shape[0]);
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        op(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        bob::sp::FFT2D op(info.shape[0], info.shape[1]);
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        op(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "FFT operation only supports 1 or 2D complex128 input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}

static object script_ifft(bob::python::const_ndarray ar) 
{
  typedef std::complex<double> dcplx;
  const bob::core::array::typeinfo& info = ar.type();
  bob::python::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        bob::sp::IFFT1D op(info.shape[0]);
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        op(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        bob::sp::IFFT2D op(info.shape[0], info.shape[1]);
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        op(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iFFT operation only supports 1 or 2D complex128 input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}

static object script_fftshift(bob::python::const_ndarray ar) 
{
  typedef std::complex<double> dcplx;
  const bob::core::array::typeinfo& info = ar.type();
  bob::python::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        bob::sp::fftshift(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        bob::sp::fftshift(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "FFTshift operation only supports 1 or 2D complex128 input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}

static object script_ifftshift(bob::python::const_ndarray ar) 
{
  typedef std::complex<double> dcplx;
  const bob::core::array::typeinfo& info = ar.type();
  bob::python::ndarray res(info);
  switch (info.nd) {
    case 1:
      {
        blitz::Array<dcplx,1> res_ = res.bz<dcplx,1>();
        bob::sp::ifftshift(ar.bz<dcplx,1>(), res_);
      }
      break;
    case 2:
      {
        blitz::Array<dcplx,2> res_ = res.bz<dcplx,2>();
        bob::sp::ifftshift(ar.bz<dcplx,2>(), res_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iFFTshift operation only supports 1 or 2D complex128 input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
  return res.self();
}

static void py_fftshift(bob::python::const_ndarray ar, bob::python::ndarray t) 
{
  const bob::core::array::typeinfo& info = ar.type();
  switch (info.nd) {
    case 1:
      {
        blitz::Array<std::complex<double>,1> t_ =
          t.bz<std::complex<double>,1>();
        bob::sp::fftshift(ar.bz<std::complex<double>,1>(), t_);
      }
      break;
    case 2:
      {
        blitz::Array<std::complex<double>,2> t_ =
          t.bz<std::complex<double>,2>();
        bob::sp::fftshift(ar.bz<std::complex<double>,2>(), t_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "FFTshift operation only supports 1 or 2D complex128 input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
}

static void py_ifftshift(bob::python::const_ndarray ar, bob::python::ndarray t) 
{
  const bob::core::array::typeinfo& info = ar.type();
  switch (info.nd) {
    case 1:
      {
        blitz::Array<std::complex<double>,1> t_ =
          t.bz<std::complex<double>,1>();
        bob::sp::ifftshift(ar.bz<std::complex<double>,1>(), t_);
      }
      break;
    case 2:
      {
        blitz::Array<std::complex<double>,2> t_ =
          t.bz<std::complex<double>,2>();
        bob::sp::ifftshift(ar.bz<std::complex<double>,2>(), t_);
      }
      break;
    default:
      PYTHON_ERROR(TypeError, "iFFTshift operation only supports 1 or 2D complex128 input arrays - you provided an array of dimensionality '" SIZE_T_FMT "'.", info.nd);
  }
}

void bind_sp_fft()
{
  // Fast Fourier Transform
  class_<bob::sp::FFT1DAbstract, boost::noncopyable>("FFT1DAbstract", "Abstract class for FFT1D", no_init)
    .def("reset", (void (bob::sp::FFT1D::*)(const size_t))&bob::sp::FFT1D::reset, (arg("self"),arg("length")), "Reset the length of the expected input signals.")
    .add_property("length", &bob::sp::FFT1D::getLength)
    ;

  class_<bob::sp::FFT1D, boost::shared_ptr<bob::sp::FFT1D>, bases<bob::sp::FFT1DAbstract> >("FFT1D", FFT1D_DOC, init<const size_t>((arg("self"), arg("length"))))
      .def(init<bob::sp::FFT1D&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_fft1d_c, (arg("self"), arg("input"), arg("output")), "Compute the FFT of the input 1D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_fft1d_p, (arg("self"), arg("input")), "Compute the FFT of the input 1D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::IFFT1D, boost::shared_ptr<bob::sp::IFFT1D>, bases<bob::sp::FFT1DAbstract> >("IFFT1D", IFFT1D_DOC, init<const size_t>((arg("self"), arg("length"))))
      .def(init<bob::sp::IFFT1D&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_ifft1d_c, (arg("self"), arg("input"), arg("output")), "Compute the inverse FFT of the input 1D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_ifft1d_p, (arg("self"), arg("input")), "Compute the inverse FFT of the input 1D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::FFT2DAbstract, boost::noncopyable>("FFT2DAbstract", "Abstract class for FFT2D", no_init)
    .def("reset", (void (bob::sp::FFT2D::*)(const size_t, const size_t))&bob::sp::FFT2D::reset, (arg("self"), arg("height"), arg("width")), "Reset the dimension of the expected input signals.")
    .add_property("height", &bob::sp::FFT2D::getHeight)
    .add_property("width", &bob::sp::FFT2D::getWidth)
    ;

  class_<bob::sp::FFT2D, boost::shared_ptr<bob::sp::FFT2D>, bases<bob::sp::FFT2DAbstract> >("FFT2D", FFT2D_DOC, init<const size_t,const size_t>((arg("self"), arg("height"), arg("width"))))
      .def(init<bob::sp::FFT2D&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_fft2d_c, (arg("self"), arg("input"), arg("output")), "Compute the FFT of the input 2D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_fft2d_p, (arg("self"), arg("input")), "Compute the FFT of the input 2D array/signal. The output is allocated and returned.")
    ;

  class_<bob::sp::IFFT2D, boost::shared_ptr<bob::sp::IFFT2D>, bases<bob::sp::FFT2DAbstract> >("IFFT2D", IFFT2D_DOC, init<const size_t,const size_t>((arg("self"), arg("height"), arg("width"))))
      .def(init<bob::sp::IFFT2D&>((arg("self"), arg("other"))))
      .def(self == self)
      .def(self != self)
      .def("__call__", &py_ifft2d_c, (arg("self"), arg("input"), arg("output")), "Compute the inverse FFT of the input 2D array/signal. The output should have the expected size and type (numpy.float64).")
      .def("__call__", &py_ifft2d_p, (arg("self"), arg("input")), "Compute the inverse FFT of the input 2D array/signal. The output is allocated and returned.")
    ;

  // fft function-like 
  def("fft", &script_fft, (arg("array")), FFT_DOC);
  def("ifft", &script_ifft, (arg("array")), IFFT_DOC);


  // fftshift
  def("fftshift", &script_fftshift, (arg("array")), FFTSHIFT_DOC);
  def("ifftshift", &script_ifftshift, (arg("array")), IFFTSHIFT_DOC);

  def("fftshift", &py_fftshift, (arg("input"),arg("output")), FFTSHIFT_DOC);
  def("ifftshift", &py_ifftshift, (arg("input"),arg("output")), IFFTSHIFT_DOC);
}
