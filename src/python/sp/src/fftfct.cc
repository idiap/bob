/**
 * @file src/python/sp/src/fft.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Binds the Fast Fourier Transform and the Fast Cosine Transform to 
 * python.
 */

#include <boost/python.hpp>

#include "core/logging.h"
#include "sp/FFT.h"
#include "sp/FCT.h"

using namespace boost::python;

static const char* FFT1_DOC = "Compute the Fast Fourier Transform of a 1D blitz array and return the result as a blitz array.";
static const char* IFFT1_DOC = "Compute the inverse Fast Fourier Transform of a 1D blitz array and return the result as a blitz array.";
static const char* FFT2_DOC = "Compute the Fast Fourier Transform of a 2D blitz array and return the result as a blitz array.";
static const char* IFFT2_DOC = "Compute the inverse Fast Fourier Transform of a 2D blitz array and return the result as a blitz array.";

static const char* FCT1_DOC = "Compute the Fast Cosine Transform of a 1D blitz array and return the result as a blitz array.";
static const char* IFCT1_DOC = "Compute the inverse Fast Cosine Transform of a 1D blitz array and return the result as a blitz array.";
static const char* FCT2_DOC = "Compute the Fast Cosine Transform of a 2D blitz array and return the result as a blitz array.";
static const char* IFCT2_DOC = "Compute the inverse Fast Cosine Transform of a 2D blitz array and return the result as a blitz array.";

static blitz::Array<std::complex<double>,1> script_fft(const blitz::Array<std::complex<double>,1>& ar)
{
  blitz::Array<std::complex<double>,1> res;
  Torch::sp::fft( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,1> script_ifft(const blitz::Array<std::complex<double>,1>& ar)
{
  blitz::Array<std::complex<double>,1> res;
  Torch::sp::ifft( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,2> script_fft(const blitz::Array<std::complex<double>,2>& ar)
{
  blitz::Array<std::complex<double>,2> res;
  Torch::sp::fft( ar, res);
  return res;
}

static blitz::Array<std::complex<double>,2> script_ifft(const blitz::Array<std::complex<double>,2>& ar)
{
  blitz::Array<std::complex<double>,2> res;
  Torch::sp::ifft( ar, res);
  return res;
}

static blitz::Array<double,1> script_fct(const blitz::Array<double,1>& ar)
{
  blitz::Array<double,1> res;
  Torch::sp::fct( ar, res);
  return res;
}

static blitz::Array<double,1> script_ifct(const blitz::Array<double,1>& ar)
{
  blitz::Array<double,1> res;
  Torch::sp::ifct( ar, res);
  return res;
}

static blitz::Array<double,2> script_fct(const blitz::Array<double,2>& ar)
{
  blitz::Array<double,2> res;
  Torch::sp::fct( ar, res);
  return res;
}

static blitz::Array<double,2> script_ifct(const blitz::Array<double,2>& ar)
{
  blitz::Array<double,2> res;
  Torch::sp::ifct( ar, res);
  return res;
}


void bind_sp_fft_fct()
{
  // Fast Fourier Transform
  def("fft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&script_fft, (arg("array")), FFT1_DOC);
  def("ifft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&script_ifft, (arg("array")), IFFT1_DOC);
  def("fft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&script_fft, (arg("array")), FFT2_DOC);
  def("ifft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&script_ifft, (arg("array")), IFFT2_DOC);
  def("fft", (void (*)(const blitz::Array<std::complex<double>,1>& ar, blitz::Array<std::complex<double>,1>& t))&Torch::sp::fft, (arg("input"),arg("output")), FFT1_DOC);
  def("ifft", (void (*)(const blitz::Array<std::complex<double>,1>& ar, blitz::Array<std::complex<double>,1>& t))&Torch::sp::ifft, (arg("input"),arg("output")), IFFT1_DOC);
  def("fft", (void (*)(const blitz::Array<std::complex<double>,2>& ar, blitz::Array<std::complex<double>,2>& t))&Torch::sp::fft, (arg("input"),arg("output")), FFT2_DOC);
  def("ifft", (void (*)(const blitz::Array<std::complex<double>,2>& ar, blitz::Array<std::complex<double>,2>& t))&Torch::sp::ifft, (arg("input"),arg("output")), IFFT2_DOC);

  // Fast Cosine Transform
  def("fct", (blitz::Array<double,1> (*)(const blitz::Array<double,1>& ar))&script_fct, (arg("array")), FCT1_DOC);
  def("ifct", (blitz::Array<double,1> (*)(const blitz::Array<double,1>& ar))&script_ifct, (arg("array")), IFCT1_DOC);
  def("fct", (blitz::Array<double,2> (*)(const blitz::Array<double,2>& ar))&script_fct, (arg("array")), FCT2_DOC);
  def("ifct", (blitz::Array<double,2> (*)(const blitz::Array<double,2>& ar))&script_ifct, (arg("array")), IFCT2_DOC);
  def("fft", (void (*)(const blitz::Array<double,1>& ar, blitz::Array<double,1>& t))&Torch::sp::fct, (arg("input"),arg("output")), FCT1_DOC);
  def("ifft", (void (*)(const blitz::Array<double,1>& ar, blitz::Array<double,1>& t))&Torch::sp::ifct, (arg("input"),arg("output")), IFCT1_DOC);
  def("fft", (void (*)(const blitz::Array<double,2>& ar, blitz::Array<double,2>& t))&Torch::sp::fct, (arg("input"),arg("output")), FCT2_DOC);
  def("ifft", (void (*)(const blitz::Array<double,2>& ar, blitz::Array<double,2>& t))&Torch::sp::ifct, (arg("input"),arg("output")), IFCT2_DOC);
}

