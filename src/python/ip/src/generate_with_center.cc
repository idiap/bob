/**
 * @file src/python/ip/src/generate_with_center.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds the generateWithCenter operation into python 
 */

#include <boost/python.hpp>

#include "ip/generateWithCenter.h"

using namespace boost::python;

static const char* GENERATEWITHCENTER2D_DOC = "Extend a 2D blitz array/image, putting a given point in the center.";
static const char* GENERATEWITHCENTER2D_MASK_DOC = "Extend a 2D blitz array/image, putting a given point in the center, taking mask into account.";
static const char* GET_GENERATEWITHCENTER_SHAPE2D_DOC = "Return the shape of the output 2D blitz array/image, when calling generateWithCenter which puts a given point of an image in the center.";
static const char* GET_GENERATEWITHCENTER_OFFSET2D_DOC = "Return the offset of the output 2D blitz array/image, when calling generateWithCenter which puts a given point of an image in the center.";


#define GENERATEWITHCENTER_DEF(T,N) \
  def("getGenerateWithCenterShape", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const int, const int))&Torch::ip::getGenerateWithCenterShape<T>, (arg("src"), arg("center_y"), arg("center_x")), GET_GENERATEWITHCENTER_SHAPE2D_DOC); \
  def("getGenerateWithCenterOffset", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const int, const int))&Torch::ip::getGenerateWithCenterOffset<T>, (arg("src"), arg("center_y"), arg("center_x")), GET_GENERATEWITHCENTER_OFFSET2D_DOC); \
  def("generateWithCenter", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&, const int, const int))&Torch::ip::generateWithCenter<T>, (arg("src"), arg("dst"), arg("center_y"), arg("center_x")), GENERATEWITHCENTER2D_DOC); \
  def("generateWithCenter", (void (*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<T,2>&, blitz::Array<bool,2>&, const int, const int))&Torch::ip::generateWithCenter<T>, (arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("center_y"), arg("center_x")), GENERATEWITHCENTER2D_MASK_DOC); \


void bind_ip_generate_with_center()
{
/*
  GENERATEWITHCENTER_DEF(bool,bool)
  GENERATEWITHCENTER_DEF(int8_t,int8)
  GENERATEWITHCENTER_DEF(int16_t,int16)
  GENERATEWITHCENTER_DEF(int32_t,int32)
  GENERATEWITHCENTER_DEF(int64_t,int64)
*/
  GENERATEWITHCENTER_DEF(uint8_t,uint8)
  GENERATEWITHCENTER_DEF(uint16_t,uint16)
/*
  GENERATEWITHCENTER_DEF(uint32_t,uint32)
  GENERATEWITHCENTER_DEF(uint64_t,uint64)
  GENERATEWITHCENTER_DEF(float,float32)
*/
  GENERATEWITHCENTER_DEF(double,float64)
/*
  GENERATEWITHCENTER_DEF(std::complex<float>,complex64)
  GENERATEWITHCENTER_DEF(std::complex<double>,complex128)
*/
}
