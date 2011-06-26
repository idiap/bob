/**
 * @file src/python/ip/src/crop_shift.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds crop and shift operations into python 
 */

#include <boost/python.hpp>

#include "ip/crop.h"
#include "ip/shift.h"

using namespace boost::python;

static const char* CROP2D_DOC = "Crop a 2D blitz array/image.";
static const char* CROP2D_MASK_DOC = "Crop a 2D blitz array/image, taking mask into account.";
static const char* CROP3D_DOC = "Crop a 3D blitz array/image.";
static const char* CROP3D_MASK_DOC = "Crop a 3D blitz array/image, taking mask into account.";
static const char* SHIFT2D_DOC = "Shift a 2D blitz array/image.";
static const char* SHIFT2D_MASK_DOC = "Shift a 2D blitz array/image, taking mask into account.";
static const char* SHIFT3D_DOC = "Shift a 3D blitz array/image.";
static const char* SHIFT3D_MASK_DOC = "Shift a 3D blitz array/image, taking mask into account.";

#define CROPSHIFT_DECL(T,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(crop_overloads_ ## N, Torch::ip::crop<T>, 6, 8) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(crop_mask_overloads_ ## N, Torch::ip::crop<T>, 8, 10) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(shift_overloads_ ## N, Torch::ip::shift<T>, 4, 6) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(shift_mask_overloads_ ## N, Torch::ip::shift<T>, 6, 8)

#define CROPSHIFT_DEF(T,N) \
  def("crop", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&, const int, const int, const int, const int, const bool, const bool))&Torch::ip::crop<T>, crop_overloads_ ## N ((arg("src"), arg("dst"), arg("crop_y"), arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false), CROP2D_DOC)); \
  def("crop", (void (*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<T,2>&, blitz::Array<bool,2>&, const int, const int, const int, const int, const bool, const bool))&Torch::ip::crop<T>, crop_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("crop_y"), arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false), CROP2D_MASK_DOC)); \
  def("crop", (void (*)(const blitz::Array<T,3>&, blitz::Array<T,3>&, const int, const int, const int, const int, const bool, const bool))&Torch::ip::crop<T>, crop_overloads_ ## N ((arg("src"), arg("dst"), arg("crop_y"), arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false), CROP3D_DOC)); \
  def("crop", (void (*)(const blitz::Array<T,3>&, const blitz::Array<bool,3>&, blitz::Array<T,3>&, blitz::Array<bool,3>&, const int, const int, const int, const int, const bool, const bool))&Torch::ip::crop<T>, crop_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("crop_y"), arg("crop_x"), arg("crop_h"), arg("crop_w"), arg("allow_out")=false, arg("zero_out")=false), CROP3D_MASK_DOC)); \
  def("shift", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,2>&, const int, const int, const bool, const bool))&Torch::ip::shift<T>, shift_overloads_ ## N ((arg("src"), arg("dst"), arg("shift_y"), arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), SHIFT2D_DOC)); \
  def("shift", (void (*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<T,2>&, blitz::Array<bool,2>&, const int, const int, const bool, const bool))&Torch::ip::shift<T>, shift_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("shift_y"), arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), SHIFT2D_MASK_DOC)); \
  def("shift", (void (*)(const blitz::Array<T,3>&, blitz::Array<T,3>&, const int, const int, const bool, const bool))&Torch::ip::shift<T>, shift_overloads_ ## N ((arg("src"), arg("dst"), arg("shift_y"), arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), SHIFT3D_DOC)); \
  def("shift", (void (*)(const blitz::Array<T,3>&, const blitz::Array<bool,3>&, blitz::Array<T,3>&, blitz::Array<bool,3>&, const int, const int, const bool, const bool))&Torch::ip::shift<T>, shift_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("shift_y"), arg("shift_x"), arg("allow_out")=false, arg("zero_out")=false), SHIFT3D_MASK_DOC)); \


/*
CROPSHIFT_DECL(bool,bool)
CROPSHIFT_DECL(int8_t,int8)
CROPSHIFT_DECL(int16_t,int16)
CROPSHIFT_DECL(int32_t,int32)
CROPSHIFT_DECL(int64_t,int64)
*/
CROPSHIFT_DECL(uint8_t,uint8)
CROPSHIFT_DECL(uint16_t,uint16)
/*
CROPSHIFT_DECL(uint32_t,uint32)
CROPSHIFT_DECL(uint64_t,uint64)
CROPSHIFT_DECL(float,float32)
*/
CROPSHIFT_DECL(double,float64)
/*
CROPSHIFT_DECL(std::complex<float>,complex64)
CROPSHIFT_DECL(std::complex<double>,complex128)
*/


void bind_ip_crop_shift()
{
/*
  CROPSHIFT_DEF(bool,bool)
  CROPSHIFT_DEF(int8_t,int8)
  CROPSHIFT_DEF(int16_t,int16)
  CROPSHIFT_DEF(int32_t,int32)
  CROPSHIFT_DEF(int64_t,int64)
*/
  CROPSHIFT_DEF(uint8_t,uint8)
  CROPSHIFT_DEF(uint16_t,uint16)
/*
  CROPSHIFT_DEF(uint32_t,uint32)
  CROPSHIFT_DEF(uint64_t,uint64)
  CROPSHIFT_DEF(float,float32)
*/
  CROPSHIFT_DEF(double,float64)
/*
  CROPSHIFT_DEF(std::complex<float>,complex64)
  CROPSHIFT_DEF(std::complex<double>,complex128)
*/
}
