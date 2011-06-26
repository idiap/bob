/**
 * @file src/python/ip/src/integral.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds integral image implementation to python 
 */

#include <boost/python.hpp>

#include "ip/integral.h"

using namespace boost::python;

static const char* INTEGRAL2D_DOC = "Compute the integral image of a 2D blitz array (image).";

template <typename U>
static inline void integral_uint8(const blitz::Array<uint8_t,2>& src, blitz::Array<U,2>& dst, const bool addZeroBorder=false)
  { Torch::ip::integral<uint8_t,U>(src,dst,addZeroBorder); }

template <typename U>
static inline void integral_uint16(const blitz::Array<uint16_t,2>& src, blitz::Array<U,2>& dst, const bool addZeroBorder=false)
  { Torch::ip::integral<uint16_t,U>(src,dst,addZeroBorder); }

template <typename U>
static inline void integral_float64(const blitz::Array<double,2>& src, blitz::Array<U,2>& dst, const bool addZeroBorder=false)
  { Torch::ip::integral<double,U>(src,dst,addZeroBorder); }

#define INTEGRAL_DECL(U,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(integral_overloads_uint8_ ## N, integral_uint8<U>, 2, 3) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(integral_overloads_uint16_ ## N, integral_uint16<U>, 2, 3) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(integral_overloads_float64_ ## N, integral_float64<U>, 2, 3)

#define INTEGRAL_DEF(U,N) \
  def(BOOST_PP_STRINGIZE(integral), (void (*)(const blitz::Array<uint8_t,2>&, blitz::Array<U,2>&, const bool))&integral_uint8<U>, integral_overloads_uint8_ ## N ((arg("src"), arg("dst"), arg("addZeroBorder")=false), INTEGRAL2D_DOC)); \
  def(BOOST_PP_STRINGIZE(integral), (void (*)(const blitz::Array<uint16_t,2>&, blitz::Array<U,2>&, const bool))&integral_uint16<U>, integral_overloads_uint16_ ## N ((arg("src"), arg("dst"), arg("addZeroBorder")=false), INTEGRAL2D_DOC)); \
  def(BOOST_PP_STRINGIZE(integral), (void (*)(const blitz::Array<double,2>&, blitz::Array<U,2>&, const bool))&integral_float64<U>, integral_overloads_float64_ ## N ((arg("src"), arg("dst"), arg("addZeroBorder")=false), INTEGRAL2D_DOC)); 


INTEGRAL_DECL(int8_t,int8)
INTEGRAL_DECL(int16_t,int16)
INTEGRAL_DECL(int32_t,int32)
INTEGRAL_DECL(int64_t,int64)
INTEGRAL_DECL(uint8_t,uint8)
INTEGRAL_DECL(uint16_t,uint16)
INTEGRAL_DECL(uint32_t,uint32)
INTEGRAL_DECL(uint64_t,uint64)
INTEGRAL_DECL(float,float32)
INTEGRAL_DECL(double,float64)


void bind_ip_integral()
{
  INTEGRAL_DEF(int8_t,int8)
  INTEGRAL_DEF(int16_t,int16)
  INTEGRAL_DEF(int32_t,int32)
  INTEGRAL_DEF(int64_t,int64)
  INTEGRAL_DEF(uint8_t,uint8)
  INTEGRAL_DEF(uint16_t,uint16)
  INTEGRAL_DEF(uint32_t,uint32)
  INTEGRAL_DEF(uint64_t,uint64)
  INTEGRAL_DEF(float,float32)
  INTEGRAL_DEF(double,float64)
}
