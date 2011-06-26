/**
 * @file src/python/ip/src/scale.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds scaling operation to python 
 */

#include <boost/python.hpp>

#include "ip/scale.h"

using namespace boost::python;

static const char* SCALEAS_DOC = "Gives back a scaled version of the original blitz array (image)";
static const char* SCALE2D_DOC = "Rescale a 2D blitz array/image with the given dimensions.";
static const char* SCALE2D_MASK_DOC = "Rescale a 2D blitz array/image with the given dimensions, taking mask into account.";


#define SCALE_DECL(T,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(scale_overloads_ ## N, Torch::ip::scale<T>, 2, 3) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(scale_mask_overloads_ ## N, Torch::ip::scale<T>, 4, 5) \

#define SCALE_DEF(T,N) \
  def("scale", (void (*)(const blitz::Array<T,2>&, blitz::Array<double,2>&, const enum Torch::ip::Rescale::Algorithm))&Torch::ip::scale<T>, scale_overloads_ ## N ((arg("src"), arg("dst"), arg("algorithm")="BilinearInterp"), SCALE2D_DOC)); \
  def("scale", (void (*)(const blitz::Array<T,2>&, const blitz::Array<bool,2>&, blitz::Array<double,2>&, blitz::Array<bool,2>&, const enum Torch::ip::Rescale::Algorithm))&Torch::ip::scale<T>, scale_mask_overloads_ ## N ((arg("src"), arg("src_mask"), arg("dst"), arg("dst_mask"), arg("algorithm")="BilinearInterp"), SCALE2D_MASK_DOC)); \
	def("scaleAs", (blitz::Array<T,2> (*)(const blitz::Array<T,2>&, const double))&Torch::ip::scaleAs<T>, (arg("original"), arg("scale_factor")), SCALEAS_DOC); \
	def("scaleAs", (blitz::Array<T,3> (*)(const blitz::Array<T,3>&, const double))&Torch::ip::scaleAs<T>, (arg("original"), arg("scale_factor")), SCALEAS_DOC); \

/*
SCALE_DECL(bool,bool)
SCALE_DECL(int8_t,int8)
SCALE_DECL(int16_t,int16)
SCALE_DECL(int32_t,int32)
SCALE_DECL(int64_t,int64)
*/
SCALE_DECL(uint8_t,uint8)
SCALE_DECL(uint16_t,uint16)
/*
SCALE_DECL(uint32_t,uint32)
SCALE_DECL(uint64_t,uint64)
SCALE_DECL(float,float32)
*/
SCALE_DECL(double,float64)
/*
SCALE_DECL(std::complex<float>,complex64)
SCALE_DECL(std::complex<double>,complex128)
*/


void bind_ip_scale()
{
  enum_<Torch::ip::Rescale::Algorithm>("RescaleAlgorithm")
    .value("NearesetNeighbour", Torch::ip::Rescale::NearestNeighbour)
    .value("BilinearInterp", Torch::ip::Rescale::BilinearInterp)
    ;

/*
  SCALE_DEF(bool,bool)
  SCALE_DEF(int8_t,int8)
  SCALE_DEF(int16_t,int16)
  SCALE_DEF(int32_t,int32)
  SCALE_DEF(int64_t,int64)
*/
  SCALE_DEF(uint8_t,uint8)
  SCALE_DEF(uint16_t,uint16)
/*
  SCALE_DEF(uint32_t,uint32)
  SCALE_DEF(uint64_t,uint64)
  SCALE_DEF(float,float32)
*/
  SCALE_DEF(double,float64)
/*
  SCALE_DEF(std::complex<float>,complex64)
  SCALE_DEF(std::complex<double>,complex128)
*/
}
