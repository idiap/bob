/**
 * @file src/python/sp/src/extrapolate.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds extrapolation to python 
 */

#include "sp/extrapolate.h"
#include "core/python/pycore.h"

using namespace boost::python;
namespace tp = Torch::python;

template <typename T, int N>
static void extrapolateConstant(const blitz::Array<T,N>& a,
    numeric::array b, T c) {
  blitz::Array<T,N> b_ = tp::numpy_bz<T,N>(b);
  Torch::sp::extrapolateConstant<T>(a, b_, c);
}

template <typename T, int N>
static void extrapolateZero(const blitz::Array<T,N>& a, numeric::array b) {
  blitz::Array<T,N> b_ = tp::numpy_bz<T,N>(b);
  Torch::sp::extrapolateZero<T>(a, b_);
}

template <typename T, int N>
static void extrapolateNearest(const blitz::Array<T,N>& a, numeric::array b) {
  blitz::Array<T,N> b_ = tp::numpy_bz<T,N>(b);
  Torch::sp::extrapolateNearest<T>(a, b_);
}

template <typename T, int N>
static void extrapolateCircular(const blitz::Array<T,N>& a, numeric::array b) {
  blitz::Array<T,N> b_ = tp::numpy_bz<T,N>(b);
  Torch::sp::extrapolateCircular<T>(a, b_);
}

template <typename T, int N>
static void extrapolateMirror(const blitz::Array<T,N>& a, numeric::array b) {
  blitz::Array<T,N> b_ = tp::numpy_bz<T,N>(b);
  Torch::sp::extrapolateMirror<T>(a, b_);
}

#define EXTRAPOLATE_DEF(T) \
  def("extrapolateConstant", &extrapolateConstant<T,1>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with a constant, given a 1D input array."); \
  def("extrapolateZero", &extrapolateZero<T,1>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with zeros, given a 1D input array."); \
  def("extrapolateNearest", &extrapolateNearest<T,1>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with nearest values, given a 1D input array."); \
  def("extrapolateCircular", &extrapolateCircular<T,1>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array using circular extrapolation, given a 1D input array."); \
  def("extrapolateMirror", &extrapolateMirror<T,1>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array using mirroring, given a 1D input array."); \
  def("extrapolateConstant", &extrapolateConstant<T,2>, (arg("src"), arg("dst")), "Extrapolates the values of a 2D array with a constant, given a 2D input array."); \
  def("extrapolateZero", &extrapolateZero<T,2>, (arg("src"), arg("dst")), "Extrapolates the values of a 2D array with zeros, given a 2D input array."); \
  def("extrapolateNearest", &extrapolateNearest<T,2>, (arg("src"), arg("dst")), "Extrapolates the values of a 2D array with nearest values, given a 2D input array."); \
  def("extrapolateCircular", &extrapolateCircular<T,2>, (arg("src"), arg("dst")), "Extrapolates the values of a 2D array using circular extrapolation, given a 2D input array."); \
  def("extrapolateMirror", &extrapolateMirror<T,2>, (arg("src"), arg("dst")), "Extrapolates the values of a 2D array using mirroring, given a 2D input array.");

void bind_sp_extrapolate()
{
  EXTRAPOLATE_DEF(bool)
  EXTRAPOLATE_DEF(int8_t)
  EXTRAPOLATE_DEF(int16_t)
  EXTRAPOLATE_DEF(int32_t)
  EXTRAPOLATE_DEF(int64_t)
  EXTRAPOLATE_DEF(uint8_t)
  EXTRAPOLATE_DEF(uint16_t)
  EXTRAPOLATE_DEF(uint32_t)
  EXTRAPOLATE_DEF(uint64_t)
  EXTRAPOLATE_DEF(float)
  EXTRAPOLATE_DEF(double)
  EXTRAPOLATE_DEF(std::complex<float>)
  EXTRAPOLATE_DEF(std::complex<double>)
}
