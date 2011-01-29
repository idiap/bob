/**
 * @file src/python/sp/src/convolution.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds convolution product to python 
 */

#include <boost/python.hpp>

#include "core/logging.h"
#include "sp/convolution.h"

using namespace boost::python;

static const char* CONVOLVE_DOC = "Compute the convolution product of two blitz arrays and return the results as a blitz array.";

#define CONVOLVE_DECL(T,N,D) static inline blitz::Array<T, D> convolve_ ## N ## _ ## D(const blitz::Array<T, D>& a, const blitz::Array<T, D>& b, const Torch::sp::ConvolutionOption option) { return Torch::sp::convolve<T>(a, b, option); } \
static inline blitz::Array<T, D> convolve_ ## N ## _ ## D ## _nopt(const blitz::Array<T, D>& a, const blitz::Array<T, D>& b) { return Torch::sp::convolve<T>(a, b); }

#define CONVOLVE_DEF(T,N,D) def("convolve", convolve_ ## N ## _ ## D, (arg("arrayA"), arg("arrayB"), arg("option")), CONVOLVE_DOC); \
def("convolve", convolve_ ## N ## _ ## D ## _nopt, (arg("arrayA"), arg("arrayB")), CONVOLVE_DOC);

CONVOLVE_DECL(bool, bool, 1)
CONVOLVE_DECL(int8_t, int8, 1)
CONVOLVE_DECL(int16_t, int16, 1)
CONVOLVE_DECL(int32_t, int32, 1)
CONVOLVE_DECL(int64_t, int64, 1)
CONVOLVE_DECL(uint8_t, uint8, 1)
CONVOLVE_DECL(uint16_t, uint16, 1)
CONVOLVE_DECL(uint32_t, uint32, 1)
CONVOLVE_DECL(uint64_t, uint64, 1)
CONVOLVE_DECL(float, float32, 1)
CONVOLVE_DECL(double, float64, 1)
CONVOLVE_DECL(std::complex<float>, complex64, 1)
CONVOLVE_DECL(std::complex<double>, complex128, 1)
CONVOLVE_DECL(bool, bool, 2)
CONVOLVE_DECL(int8_t, int8, 2)
CONVOLVE_DECL(int16_t, int16, 2)
CONVOLVE_DECL(int32_t, int32, 2)
CONVOLVE_DECL(int64_t, int64, 2)
CONVOLVE_DECL(uint8_t, uint8, 2)
CONVOLVE_DECL(uint16_t, uint16, 2)
CONVOLVE_DECL(uint32_t, uint32, 2)
CONVOLVE_DECL(uint64_t, uint64, 2)
CONVOLVE_DECL(float, float32, 2)
CONVOLVE_DECL(double, float64, 2)
CONVOLVE_DECL(std::complex<float>, complex64, 2)
CONVOLVE_DECL(std::complex<double>, complex128, 2)

void bind_sp_convolution()
{
  enum_<Torch::sp::ConvolutionOption>("ConvolutionOption")
    .value("FULL", Torch::sp::FULL)
    .value("SAME", Torch::sp::SAME)
    .value("VALID", Torch::sp::VALID)
    ;
 
  CONVOLVE_DEF(bool, bool, 1)
  CONVOLVE_DEF(int8_t, int8, 1)
  CONVOLVE_DEF(int16_t, int16, 1)
  CONVOLVE_DEF(int32_t, int32, 1)
  CONVOLVE_DEF(int64_t, int64, 1)
  CONVOLVE_DEF(uint8_t, uint8, 1)
  CONVOLVE_DEF(uint16_t, uint16, 1)
  CONVOLVE_DEF(uint32_t, uint32, 1)
  CONVOLVE_DEF(uint64_t, uint64, 1)
  CONVOLVE_DEF(float, float32, 1)
  CONVOLVE_DEF(double, float64, 1)
  CONVOLVE_DEF(std::complex<float>, complex64, 1)
  CONVOLVE_DEF(std::complex<double>, complex128, 1)
  CONVOLVE_DEF(bool, bool, 2)
  CONVOLVE_DEF(int8_t, int8, 2)
  CONVOLVE_DEF(int16_t, int16, 2)
  CONVOLVE_DEF(int32_t, int32, 2)
  CONVOLVE_DEF(int64_t, int64, 2)
  CONVOLVE_DEF(uint8_t, uint8, 2)
  CONVOLVE_DEF(uint16_t, uint16, 2)
  CONVOLVE_DEF(uint32_t, uint32, 2)
  CONVOLVE_DEF(uint64_t, uint64, 2)
  CONVOLVE_DEF(float, float32, 2)
  CONVOLVE_DEF(double, float64, 2)
  CONVOLVE_DEF(std::complex<float>, complex64, 2)
  CONVOLVE_DEF(std::complex<double>, complex128, 2)
}

