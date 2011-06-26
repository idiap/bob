/**
 * @file src/python/ip/src/flipflop.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds the flip and flop operations into python 
 */

#include <boost/python.hpp>

#include "ip/flipflop.h"

using namespace boost::python;

static const char* FLIP2D_DOC = "Flip a 2D blitz array/image upside-down.";
static const char* FLIP3D_DOC = "Flip a 3D blitz array/image upside-down.";
static const char* FLOP2D_DOC = "Flop a 2D blitz array/image left-right.";
static const char* FLOP3D_DOC = "Flop a 3D blitz array/image left-right.";


#define FLIPFLOP_DEF(T,N) \
  def("flip", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&Torch::ip::flip<T>, (arg("src"), arg("dst")), FLIP2D_DOC); \
  def("flip", (void (*)(const blitz::Array<T,3>&, blitz::Array<T,3>&))&Torch::ip::flip<T>, (arg("src"), arg("dst")), FLIP3D_DOC); \
  def("flop", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&))&Torch::ip::flop<T>, (arg("src"), arg("dst")), FLOP2D_DOC); \
  def("flop", (void (*)(const blitz::Array<T,3>&, blitz::Array<T,3>&))&Torch::ip::flop<T>, (arg("src"), arg("dst")), FLOP3D_DOC); \


void bind_ip_flipflop()
{
/*
  FLIPFLOP_DEF(bool,bool)
  FLIPFLOP_DEF(int8_t,int8)
  FLIPFLOP_DEF(int16_t,int16)
  FLIPFLOP_DEF(int32_t,int32)
  FLIPFLOP_DEF(int64_t,int64)
*/
  FLIPFLOP_DEF(uint8_t,uint8)
  FLIPFLOP_DEF(uint16_t,uint16)
/*
  FLIPFLOP_DEF(uint32_t,uint32)
  FLIPFLOP_DEF(uint64_t,uint64)
  FLIPFLOP_DEF(float,float32)
*/
  FLIPFLOP_DEF(double,float64)
/*
  FLIPFLOP_DEF(std::complex<float>,complex64)
  FLIPFLOP_DEF(std::complex<double>,complex128)
*/
}
