/**
 * @file src/python/ip/src/extrapolate_mask.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds the extrapolateMask operation into python 
 */

#include <boost/python.hpp>

#include "ip/extrapolateMask.h"

using namespace boost::python;

static const char* EXTRAPOLATE2D_MASK_DOC = "Extrapolate a 2D blitz array/image, taking mask into account.";

#define EXTRAPOLATEMASK_DEF(T,N) \
  def("extrapolateMask", (void (*)(const blitz::Array<bool,2>&, blitz::Array<T,2>&))&Torch::ip::extrapolateMask<T>, (arg("src_mask"), arg("img")), EXTRAPOLATE2D_MASK_DOC); \


void bind_ip_extrapolate_mask()
{
/*
  EXTRAPOLATEMASK_DEF(bool,bool)
  EXTRAPOLATEMASK_DEF(int8_t,int8)
  EXTRAPOLATEMASK_DEF(int16_t,int16)
  EXTRAPOLATEMASK_DEF(int32_t,int32)
  EXTRAPOLATEMASK_DEF(int64_t,int64)
*/
  EXTRAPOLATEMASK_DEF(uint8_t,uint8)
  EXTRAPOLATEMASK_DEF(uint16_t,uint16)
/*
  EXTRAPOLATEMASK_DEF(uint32_t,uint32)
  EXTRAPOLATEMASK_DEF(uint64_t,uint64)
  EXTRAPOLATEMASK_DEF(float,float32)
*/
  EXTRAPOLATEMASK_DEF(double,float64)
/*
  EXTRAPOLATEMASK_DEF(std::complex<float>,complex64)
  EXTRAPOLATEMASK_DEF(std::complex<double>,complex128)
*/
}
