/**
 * @file src/python/sp/src/convolution.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds convolution product to python 
 */

#include <boost/python.hpp>

#include "core/logging.h"
#include "sp/convolution.h"

#include "core/python/pycore.h"

using namespace boost::python;
namespace tp = Torch::python;

static const char* CONVOLVE_DOC = "Compute the convolution product of two blitz arrays using zero padding and return the results as a blitz array. The option field allows to give information about the size of the output (FULL, SAME, VALID)";

template <typename T, int N>
static void py_convolve(const blitz::Array<T,N>& a, const blitz::Array<T,N>& b, numeric::array c, const enum Torch::sp::Convolution::SizeOption d=Torch::sp::Convolution::Full, const enum Torch::sp::Convolution::BorderOption e=Torch::sp::Convolution::Zero) {
  blitz::Array<T,N> c_ = tp::numpy_bz<T,N>(c);
  Torch::sp::convolve(a, b, c_, d, e);
}

template <typename T, int N>
static void py_convolveSep(const blitz::Array<T,N>& a, const blitz::Array<T,1>& b, numeric::array c, const int d, const enum Torch::sp::Convolution::SizeOption e=Torch::sp::Convolution::Full, const enum Torch::sp::Convolution::BorderOption f=Torch::sp::Convolution::Zero) {
  blitz::Array<T,N> c_ = tp::numpy_bz<T,N>(c);
  Torch::sp::convolveSep(a, b, c_, d, e, f);
}

#define CONVOLVE_DECL(T,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(getConvolveOutputSize_overloads_ ## N, Torch::sp::getConvolveOutputSize<T>, 2, 3) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(getConvolveSepOutputSize_overloads_ ## N, Torch::sp::getConvolveSepOutputSize<T>, 3, 4) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_overloads_1_ ## N, (py_convolve<T,1>), 3, 5) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(convolve_overloads_2_ ## N, (py_convolve<T,2>), 3, 5) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(convolveSep_overloads_2_ ## N, (py_convolveSep<T,2>), 4, 6) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(convolveSep_overloads_3_ ## N, (py_convolveSep<T,3>), 4, 6) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(convolveSep_overloads_4_ ## N, (py_convolveSep<T,4>), 4, 6)

#define CONVOLVE_DEF(T,N) \
  def("getConvolveOutputSize", (const blitz::TinyVector<int,1> (*)(const blitz::Array<T,1>&, const blitz::Array<T,1>&, const enum Torch::sp::Convolution::SizeOption))&Torch::sp::getConvolveOutputSize<T>, getConvolveOutputSize_overloads_ ## N ((arg("b"), arg("c"), arg("size_opt")="Full"), "Gets the required size of the result of a 1D convolution product")); \
  def("getConvolveOutputSize", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const blitz::Array<T,2>&, const enum Torch::sp::Convolution::SizeOption))&Torch::sp::getConvolveOutputSize<T>, getConvolveOutputSize_overloads_ ## N ((arg("b"), arg("c"), arg("size_opt")="Full"), "Gets the required size of the result of a 2D convolution product")); \
  def("getConvolveSepOutputSize", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const blitz::Array<T,1>&, const int, const enum Torch::sp::Convolution::SizeOption))&Torch::sp::getConvolveSepOutputSize<T>, getConvolveSepOutputSize_overloads_ ## N ((arg("b"), arg("c"), arg("dim"), arg("size_opt")="Full"), "Gets the required size of the result of a 2D separable convolution product")); \
  def("getConvolveSepOutputSize", (const blitz::TinyVector<int,3> (*)(const blitz::Array<T,3>&, const blitz::Array<T,1>&, const int, const enum Torch::sp::Convolution::SizeOption))&Torch::sp::getConvolveSepOutputSize<T>, getConvolveSepOutputSize_overloads_ ## N ((arg("b"), arg("c"), arg("dim"), arg("size_opt")="Full"), "Gets the required size of the result of a 3D separable convolution product")); \
  def("getConvolveSepOutputSize", (const blitz::TinyVector<int,4> (*)(const blitz::Array<T,4>&, const blitz::Array<T,1>&, const int, const enum Torch::sp::Convolution::SizeOption))&Torch::sp::getConvolveSepOutputSize<T>, getConvolveSepOutputSize_overloads_ ## N ((arg("b"), arg("c"), arg("dim"), arg("size_opt")="Full"), "Gets the required size of the result of a 4D separable convolution product")); \
  def("convolve", &py_convolve<T,1>, convolve_overloads_1_ ## N ((arg("b"), arg("c"), arg("a"), arg("size_opt")="Full", arg("border_opt")="Zero"), CONVOLVE_DOC)); \
  def("convolve", &py_convolve<T,2>, convolve_overloads_2_ ## N ((arg("b"), arg("c"), arg("a"), arg("size_opt")="Full", arg("border_opt")="Zero"), CONVOLVE_DOC)); \
  def("convolveSep", &py_convolveSep<T,2>, convolveSep_overloads_2_ ## N ((arg("b"), arg("c"), arg("a"), arg("dim"), arg("size_opt")="Full", arg("border_opt")="Zero"), "Computes the convolution product of a 2D blitz array with a 1D one, along the given dimension. (useful for separable convolution)")); \
  def("convolveSep", &py_convolveSep<T,3>, convolveSep_overloads_3_ ## N ((arg("b"), arg("c"), arg("a"), arg("dim"), arg("size_opt")="Full", arg("border_opt")="Zero"), "Computes the convolution product of a 3D blitz array with a 1D one, along the given dimension. (useful for separable convolution)")); \
  def("convolveSep", &py_convolveSep<T,4>, convolveSep_overloads_4_ ## N ((arg("b"), arg("c"), arg("a"), arg("dim"), arg("size_opt")="Full", arg("border_opt")="Zero"), "Computes the convolution product of a 4D blitz array with a 1D one, along the given dimension. (useful for separable convolution)"));

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

