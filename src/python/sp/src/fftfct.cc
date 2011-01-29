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

void bind_sp_fft_fct()
{
  // Fast Fourier Transfomr
  def("fft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&Torch::sp::fft, (arg("array")), FFT1_DOC);
  def("ifft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&Torch::sp::ifft, (arg("array")), IFFT1_DOC);
  def("fft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&Torch::sp::fft, (arg("array")), FFT2_DOC);
  def("ifft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&Torch::sp::ifft, (arg("array")), IFFT2_DOC);

  // Fast Cosine Transform
  def("fct", (blitz::Array<double,1> (*)(const blitz::Array<double,1>& ar))&Torch::sp::fct, (arg("array")), FCT1_DOC);
  def("ifct", (blitz::Array<double,1> (*)(const blitz::Array<double,1>& ar))&Torch::sp::ifct, (arg("array")), IFCT1_DOC);
  def("fct", (blitz::Array<double,2> (*)(const blitz::Array<double,2>& ar))&Torch::sp::fct, (arg("array")), FCT2_DOC);
  def("ifct", (blitz::Array<double,2> (*)(const blitz::Array<double,2>& ar))&Torch::sp::ifct, (arg("array")), IFCT2_DOC);

}

