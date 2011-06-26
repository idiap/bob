/**
 * @file src/python/ip/src/shear.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds shearing operation into python 
 */

#include <boost/python.hpp>

#include "ip/shear.h"

using namespace boost::python;

static const char* GET_SHEARX_SHAPE2D_DOC = "Return the shape of the output 2D blitz array/image, when calling shearX.";
static const char* GET_SHEARY_SHAPE2D_DOC = "Return the shape of the output 2D blitz array/image, when calling shearY.";
static const char* SHEARX2D_DOC = "Shear a 2D blitz array/image with the given shear parameter along the X-dimension.";
static const char* SHEARY2D_DOC = "Shear a 2D blitz array/image with the given shear parameter along the Y-dimension.";
static const char* SHEARX2D_MASK_DOC = "Shear a 2D blitz array/image with the given shear parameter along the X-dimension, taking mask into account.";
static const char* SHEARY2D_MASK_DOC = "Shear a 2D blitz array/image with the given shear parameter along the Y-dimension, taking mask into account.";


#define SHEAR_DECL(T,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(shearX_overloads_ ## N, Torch::ip::shearX<T>, 3, 4) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(shearX_mask_overloads_ ## N, Torch::ip::shearX<T>, 5, 6) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(shearY_overloads_ ## N, Torch::ip::shearY<T>, 3, 4) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(shearY_mask_overloads_ ## N, Torch::ip::shearY<T>, 5, 6) 

#define SHEAR_DEF(T,N) \
  def("getShearXShape", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const double))&Torch::ip::getShearXShape<T>, (arg("src"), arg("shear")), GET_SHEARX_SHAPE2D_DOC); \
  def("getShearYShape", (const blitz::TinyVector<int,2> (*)(const blitz::Array<T,2>&, const double))&Torch::ip::getShearYShape<T>, (arg("src"), arg("shear")), GET_SHEARY_SHAPE2D_DOC); \
  def("shearX", (void (*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const double, const bool))&Torch::ip::shearX<T>, shearX_overloads_ ## N ((arg("src"), arg("dst"), arg("angle"), arg("antialias")=true), SHEARX2D_DOC)); \
  def("shearY", (void (*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const double, const bool))&Torch::ip::shearY<T>, shearY_overloads_ ## N ((arg("src"), arg("dst"), arg("angle"), arg("antialias")=true), SHEARY2D_DOC)); \
  def("shearX", (void (*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<double,2>&, blitz::Array<bool,2>&, const double, const bool))&Torch::ip::shearX<T>, shearX_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("angle"), arg("antialias")=true), SHEARX2D_MASK_DOC)); \
  def("shearY", (void (*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<double,2>&, blitz::Array<bool,2>&, const double, const bool))&Torch::ip::shearY<T>, shearY_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("angle"), arg("antialias")=true), SHEARY2D_MASK_DOC)); \

/*
SHEAR_DECL(bool,bool)
SHEAR_DECL(int8_t,int8)
SHEAR_DECL(int16_t,int16)
SHEAR_DECL(int32_t,int32)
SHEAR_DECL(int64_t,int64)
*/
SHEAR_DECL(uint8_t,uint8)
SHEAR_DECL(uint16_t,uint16)
/*
SHEAR_DECL(uint32_t,uint32)
SHEAR_DECL(uint64_t,uint64)
SHEAR_DECL(float,float32)
*/
SHEAR_DECL(double,float64)
/*
SHEAR_DECL(std::complex<float>,complex64)
SHEAR_DECL(std::complex<double>,complex128)
*/


void bind_ip_shear()
{
/*
  SHEAR_DEF(bool,bool)
  SHEAR_DEF(int8_t,int8)
  SHEAR_DEF(int16_t,int16)
  SHEAR_DEF(int32_t,int32)
  SHEAR_DEF(int64_t,int64)
*/
  SHEAR_DEF(uint8_t,uint8)
  SHEAR_DEF(uint16_t,uint16)
/*
  SHEAR_DEF(uint32_t,uint32)
  SHEAR_DEF(uint64_t,uint64)
  SHEAR_DEF(float,float32)
*/
  SHEAR_DEF(double,float64)
/*
  SHEAR_DEF(std::complex<float>,complex64)
  SHEAR_DEF(std::complex<double>,complex128)
*/
}
