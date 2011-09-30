/**
 * @file src/python/sp/src/fft_dct.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Fast Fourier Transform and the (fast) Discrete Cosine 
 * Transform to python.
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

#include "core/python/pycore.h"

using namespace boost::python;
namespace sp = Torch::sp;
namespace tp = Torch::python;

static const char* FFT1D_DOC = "Objects of this class, after configuration, can compute the direct FFT of a 1D array/signal.";
static const char* IFFT1D_DOC = "Objects of this class, after configuration, can compute the inverse FFT of a 1D array/signal.";
static const char* FFT2D_DOC = "Objects of this class, after configuration, can compute the direct FFT of a 2D array/signal.";
static const char* IFFT2D_DOC = "Objects of this class, after configuration, can compute the inverse FFT of a 2D array/signal.";

static const char* DCT1D_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 1D array/signal.";
static const char* IDCT1D_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 1D array/signal.";
static const char* DCT2D_DOC = "Objects of this class, after configuration, can compute the direct DCT of a 2D array/signal.";
static const char* IDCT2D_DOC = "Objects of this class, after configuration, can compute the inverse DCT of a 2D array/signal.";

static const char* FFT1_DOC = "Compute the direct FFT of a 1D array/signal.";
static const char* IFFT1_DOC = "Compute the inverse FFT of a 1D array/signal.";
static const char* FFT2_DOC = "Compute the direct FFT of a 2D array/signal.";
static const char* IFFT2_DOC = "Compute the inverse FFT of a 2D array/signal.";

static const char* DCT1_DOC = "Compute the direct DCT of a 1D array/signal.";
static const char* IDCT1_DOC = "Compute the inverse DCT of a 1D array/signal.";
static const char* DCT2_DOC = "Compute the direct DCT of a 2D array/signal.";
static const char* IDCT2_DOC = "Compute the inverse DCT of a 2D array/signal.";

static const char* FFTSHIFT1_DOC = "Inverse the two halves of a 1D blitz array and return the result as a blitz array.";
static const char* IFFTSHIFT1_DOC = "Inverse the two halves of a 1D blitz array and return the result as a blitz array. Calling iffshift over an fftshifted array should give the original array back.";
static const char* FFTSHIFT2_DOC = "Swap the four quadrants of a 2D blitz array and return the result as a blitz array.";
static const char* IFFTSHIFT2_DOC = "Swap the four quadrants of a 2D blitz array and return the result as a blitz array. Calling iffshift over an fftshifted array should give the original array back.";


static blitz::Array<std::complex<double>,1> script_fft(const blitz::Array<std::complex<double>,1>& ar)
{
  Torch::sp::FFT1D fft(ar.extent(0));
  blitz::Array<std::complex<double>,1> res(ar.shape());
  fft( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,1> script_ifft(const blitz::Array<std::complex<double>,1>& ar)
{
  Torch::sp::IFFT1D ifft(ar.extent(0));
  blitz::Array<std::complex<double>,1> res(ar.shape());
  ifft( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,2> script_fft(const blitz::Array<std::complex<double>,2>& ar)
{
  Torch::sp::FFT2D fft(ar.extent(0),ar.extent(1));
  blitz::Array<std::complex<double>,2> res(ar.shape());
  fft( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,2> script_ifft(const blitz::Array<std::complex<double>,2>& ar)
{
  Torch::sp::IFFT2D ifft(ar.extent(0),ar.extent(1));
  blitz::Array<std::complex<double>,2> res(ar.shape());
  ifft( ar, res);
  return res;
}

static blitz::Array<double,1> script_dct(const blitz::Array<double,1>& ar)
{
  Torch::sp::DCT1D dct(ar.extent(0));
  blitz::Array<double,1> res(ar.shape());
  dct( ar, res);
  return res;
}

static blitz::Array<double,1> script_idct(const blitz::Array<double,1>& ar)
{
  Torch::sp::IDCT1D idct(ar.extent(0));
  blitz::Array<double,1> res(ar.shape());
  idct( ar, res);
  return res;
}

static blitz::Array<double,2> script_dct(const blitz::Array<double,2>& ar)
{
  Torch::sp::DCT2D dct(ar.extent(0),ar.extent(1));
  blitz::Array<double,2> res(ar.shape());
  dct( ar, res);
  return res;
}

static blitz::Array<double,2> script_idct(const blitz::Array<double,2>& ar)
{
  Torch::sp::IDCT2D idct(ar.extent(0),ar.extent(1));
  blitz::Array<double,2> res(ar.shape());
  idct( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,1> script_fftshift(const blitz::Array<std::complex<double>,1>& ar)
{
  blitz::Array<std::complex<double>,1> res(ar.shape());
  Torch::sp::fftshift( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,1> script_ifftshift(const blitz::Array<std::complex<double>,1>& ar)
{
  blitz::Array<std::complex<double>,1> res(ar.shape());
  Torch::sp::ifftshift( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,2> script_fftshift(const blitz::Array<std::complex<double>,2>& ar)
{
  blitz::Array<std::complex<double>,2> res(ar.shape());
  Torch::sp::fftshift( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,2> script_ifftshift(const blitz::Array<std::complex<double>,2>& ar)
{
  blitz::Array<std::complex<double>,2> res(ar.shape());
  Torch::sp::ifftshift( ar, res);
  return res;
}

static void py_fft1d_call(sp::FFT1D& a, const blitz::Array<std::complex<double>,1>& b, numeric::array c) {
  blitz::Array<std::complex<double>,1> c_ =
    tp::numpy_bz<std::complex<double>,1>(c);
  a(b, c_);
}

static void py_ifft1d_call(sp::IFFT1D& a, const blitz::Array<std::complex<double>,1>& b, numeric::array c) {
  blitz::Array<std::complex<double>,1> c_ =
    tp::numpy_bz<std::complex<double>,1>(c);
  a(b, c_);
}

static void py_fft2d_call(sp::FFT2D& a, const blitz::Array<std::complex<double>,2>& b, numeric::array c) {
  blitz::Array<std::complex<double>,2> c_ =
    tp::numpy_bz<std::complex<double>,2>(c);
  a(b, c_);
}

static void py_ifft2d_call(sp::IFFT2D& a, const blitz::Array<std::complex<double>,2>& b, numeric::array c) {
  blitz::Array<std::complex<double>,2> c_ =
    tp::numpy_bz<std::complex<double>,2>(c);
  a(b, c_);
}

static void py_dct1d_call(sp::DCT1D& a, const blitz::Array<double,1>& b, numeric::array c) {
  blitz::Array<double,1> c_ = tp::numpy_bz<double,1>(c);
  a(b, c_);
}

static void py_idct1d_call(sp::IDCT1D& a, const blitz::Array<double,1>& b, numeric::array c) {
  blitz::Array<double,1> c_ = tp::numpy_bz<double,1>(c);
  a(b, c_);
}

static void py_dct2d_call(sp::DCT2D& a, const blitz::Array<double,2>& b, numeric::array c) {
  blitz::Array<double,2> c_ = tp::numpy_bz<double,2>(c);
  a(b, c_);
}

static void py_idct2d_call(sp::IDCT2D& a, const blitz::Array<double,2>& b, numeric::array c) {
  blitz::Array<double,2> c_ = tp::numpy_bz<double,2>(c);
  a(b, c_);
}

template <int N>
static void py_fftshift(const blitz::Array<std::complex<double>,N>& ar, 
    numeric::array t) {
  blitz::Array<std::complex<double>,N> t_ = 
    tp::numpy_bz<std::complex<double>,N>(t);
  Torch::sp::fftshift(ar, t_);
}

template <int N>
static void py_ifftshift(const blitz::Array<std::complex<double>,N>& ar, 
    numeric::array t) {
  blitz::Array<std::complex<double>,N> t_ = 
    tp::numpy_bz<std::complex<double>,N>(t);
  Torch::sp::ifftshift(ar, t_);
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
  def("fft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&script_fft, (arg("array")), FFT1_DOC);
  def("ifft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&script_ifft, (arg("array")), IFFT1_DOC);
  def("fft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&script_fft, (arg("array")), FFT2_DOC);
  def("ifft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&script_ifft, (arg("array")), IFFT2_DOC);

  def("dct", (blitz::Array<double,1> (*)(const blitz::Array<double,1>& ar))&script_dct, (arg("array")), DCT1_DOC);
  def("idct", (blitz::Array<double,1> (*)(const blitz::Array<double,1>& ar))&script_idct, (arg("array")), IDCT1_DOC);
  def("dct", (blitz::Array<double,2> (*)(const blitz::Array<double,2>& ar))&script_dct, (arg("array")), DCT2_DOC);
  def("idct", (blitz::Array<double,2> (*)(const blitz::Array<double,2>& ar))&script_idct, (arg("array")), IDCT2_DOC);


  // fftshift
  def("fftshift", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&script_fftshift, (arg("array")), FFTSHIFT1_DOC);
  def("ifftshift", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&script_ifftshift, (arg("array")), IFFTSHIFT1_DOC);
  def("fftshift", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&script_fftshift, (arg("array")), FFTSHIFT2_DOC);
  def("ifftshift", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&script_ifftshift, (arg("array")), IFFTSHIFT2_DOC);
  def("fftshift", &py_fftshift<1>, (arg("input"),arg("output")), FFTSHIFT1_DOC);
  def("ifftshift", &py_ifftshift<1>, (arg("input"),arg("output")), IFFTSHIFT1_DOC);
  def("fftshift", &py_fftshift<2>, (arg("input"),arg("output")), FFTSHIFT2_DOC);
  def("ifftshift", &py_ifftshift<2>, (arg("input"),arg("output")), IFFTSHIFT2_DOC);
}

