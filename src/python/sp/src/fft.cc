/**
 * @file src/python/sp/src/fft.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds the Fast Fourier Transform to python 
 */

#include <boost/python.hpp>

#include "core/logging.h"
#include "sp/FFT.h"

using namespace boost::python;

static const char* FFT1_DOC = "Compute the Fast Fourier Transform of a 1D blitz array and return the result as a blitz array.";
static const char* IFFT1_DOC = "Compute the inverse Fast Fourier Transform of a 1D blitz array and return the result as a blitz array.";
static const char* FFT2_DOC = "Compute the Fast Fourier Transform of a 2D blitz array and return the result as a blitz array.";
static const char* IFFT2_DOC = "Compute the inverse Fast Fourier Transform of a 2D blitz array and return the result as a blitz array.";

void bind_sp_fft()
{
  def("fft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&Torch::sp::fft, (arg("array")), FFT1_DOC);
  def("ifft", (blitz::Array<std::complex<double>,1> (*)(const blitz::Array<std::complex<double>,1>& ar))&Torch::sp::ifft, (arg("array")), IFFT1_DOC);
  def("fft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&Torch::sp::fft, (arg("array")), FFT2_DOC);
  def("ifft", (blitz::Array<std::complex<double>,2> (*)(const blitz::Array<std::complex<double>,2>& ar))&Torch::sp::ifft, (arg("array")), IFFT2_DOC);
}

