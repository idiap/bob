/**
 * @file src/python/ip/src/gamma_correction.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds gamma correction into python 
 */

#include <boost/python.hpp>

#include "ip/gammaCorrection.h"

using namespace boost::python;

static const char* GAMMACORRECTION2D_DOC = "Perform a power-law gamma correction on a 2D blitz array/image.";


#define GAMMACORRECTION_DEF(T,N) \
  def("gammaCorrection", (void (*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const double))&Torch::ip::gammaCorrection<T>, (arg("src"), arg("dst"), arg("gamma")), GAMMACORRECTION2D_DOC); \

void bind_ip_gamma_correction()
{
/*
  GAMMACORRECTION_DEF(bool,bool)
  GAMMACORRECTION_DEF(int8_t,int8)
  GAMMACORRECTION_DEF(int16_t,int16)
  GAMMACORRECTION_DEF(int32_t,int32)
  GAMMACORRECTION_DEF(int64_t,int64)
*/
  GAMMACORRECTION_DEF(uint8_t,uint8)
  GAMMACORRECTION_DEF(uint16_t,uint16)
/*
  GAMMACORRECTION_DEF(uint32_t,uint32)
  GAMMACORRECTION_DEF(uint64_t,uint64)
  GAMMACORRECTION_DEF(float,float32)
*/
  GAMMACORRECTION_DEF(double,float64)
/*
  GAMMACORRECTION_DEF(std::complex<float>,complex64)
  GAMMACORRECTION_DEF(std::complex<double>,complex128)
*/
}
