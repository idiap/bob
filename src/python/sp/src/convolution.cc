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

static const char* CONVOLVE_DOC = "Compute the convolution product of two blitz arrays using zero padding and return the results as a blitz array. The option field allows to give information about the size of the output (FULL, SAME, VALID)";

#define CONVOLVE_DECL(T,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_overloads_ ## N, Torch::sp::convolve<T>, 3, 5)

#define CONVOLVE_DEF(T,N) \
  def("convolve", (void (*)(const blitz::Array<T,1>&, const blitz::Array<T,1>&, blitz::Array<T,1>&, const enum Torch::sp::Convolution::SizeOption, const enum Torch::sp::Convolution::BorderOption))&Torch::sp::convolve<T>, convolve_overloads_ ## N ((arg("b"), arg("c"), arg("a"), arg("size_opt")="Full", arg("border_opt")="Zero"), CONVOLVE_DOC)); \
  def("convolve", (void (*)(const blitz::Array<T,2>&, const blitz::Array<T,2>&, blitz::Array<T,2>&, const enum Torch::sp::Convolution::SizeOption, const enum Torch::sp::Convolution::BorderOption))&Torch::sp::convolve<T>, convolve_overloads_ ## N ((arg("b"), arg("c"), arg("a"), arg("size_opt")="Full", arg("border_opt")="Zero"), CONVOLVE_DOC));

CONVOLVE_DECL(bool,bool)
CONVOLVE_DECL(int8_t,int8)
CONVOLVE_DECL(int16_t,int16)
CONVOLVE_DECL(int32_t,int32)
CONVOLVE_DECL(int64_t,int64)
CONVOLVE_DECL(uint8_t,uint8)
CONVOLVE_DECL(uint16_t,uint16)
CONVOLVE_DECL(uint32_t,uint32)
CONVOLVE_DECL(uint64_t,uint64)
CONVOLVE_DECL(float,float32)
CONVOLVE_DECL(double,float64)
CONVOLVE_DECL(std::complex<float>,complex64)
CONVOLVE_DECL(std::complex<double>,complex128)

void bind_sp_convolution()
{
  enum_<Torch::sp::Convolution::SizeOption>("ConvolutionSize")
    .value("Full", Torch::sp::Convolution::Full)
    .value("Same", Torch::sp::Convolution::Same)
    .value("Valid", Torch::sp::Convolution::Valid)
    ;
 
  enum_<Torch::sp::Convolution::BorderOption>("ConvolutionBorder")
    .value("Zero", Torch::sp::Convolution::Zero)
    .value("NearestNeighbour", Torch::sp::Convolution::NearestNeighbour)
    .value("Circular", Torch::sp::Convolution::Circular)
    .value("Mirror", Torch::sp::Convolution::Mirror)
    ;
 
  CONVOLVE_DEF(bool,bool)
  CONVOLVE_DEF(int8_t,int8)
  CONVOLVE_DEF(int16_t,int16)
  CONVOLVE_DEF(int32_t,int32)
  CONVOLVE_DEF(int64_t,int64)
  CONVOLVE_DEF(uint8_t,uint8)
  CONVOLVE_DEF(uint16_t,uint16)
  CONVOLVE_DEF(uint32_t,uint32)
  CONVOLVE_DEF(uint64_t,uint64)
  CONVOLVE_DEF(float,float32)
  CONVOLVE_DEF(double,float64)
  CONVOLVE_DEF(std::complex<float>,complex64)
  CONVOLVE_DEF(std::complex<double>,complex128)
}

