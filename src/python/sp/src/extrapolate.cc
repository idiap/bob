/**
 * @file src/python/sp/src/extrapolate.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds extrapolation to python 
 */

#include <boost/python.hpp>
#include "sp/extrapolate.h"

using namespace boost::python;

#define EXTRAPOLATE_DEF(T,N) \
  def("extrapolateConstant", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&, const T))&Torch::sp::extrapolateConstant<T>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with a constant, given a 1D input array."); \
  def("extrapolateZero", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&))&Torch::sp::extrapolateZero<T>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with zeros, given a 1D input array."); \
  def("extrapolateNearest", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&))&Torch::sp::extrapolateNearest<T>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array with nearest values, given a 1D input array."); \
  def("extrapolateCircular", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&))&Torch::sp::extrapolateCircular<T>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array using circular extrapolation, given a 1D input array."); \
  def("extrapolateMirror", (void (*)(const blitz::Array<T,1>&, blitz::Array<T,1>&))&Torch::sp::extrapolateMirror<T>, (arg("src"), arg("dst")), "Extrapolates the values of a 1D array using mirroring, given a 1D input array.");

void bind_sp_extrapolate()
{
  EXTRAPOLATE_DEF(bool,bool)
  EXTRAPOLATE_DEF(int8_t,int8)
  EXTRAPOLATE_DEF(int16_t,int16)
  EXTRAPOLATE_DEF(int32_t,int32)
  EXTRAPOLATE_DEF(int64_t,int64)
  EXTRAPOLATE_DEF(uint8_t,uint8)
  EXTRAPOLATE_DEF(uint16_t,uint16)
  EXTRAPOLATE_DEF(uint32_t,uint32)
  EXTRAPOLATE_DEF(uint64_t,uint64)
  EXTRAPOLATE_DEF(float,float32)
  EXTRAPOLATE_DEF(double,float64)
  EXTRAPOLATE_DEF(std::complex<float>,complex64)
  EXTRAPOLATE_DEF(std::complex<double>,complex128)
}

