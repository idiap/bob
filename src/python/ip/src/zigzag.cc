/**
 * @file src/python/ip/src/zigzag.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds the zigzag operation into python 
 */

#include <boost/python.hpp>

#include "ip/zigzag.h"

using namespace boost::python;

static const char* ZIGZAG2D_DOC = "Extract a 1D blitz array using a zigzag pattern from a 2D blitz array/image.";


#define ZIGZAG_DECL(T,N) \
  BOOST_PYTHON_FUNCTION_OVERLOADS(zigzag_overloads_ ## N, Torch::ip::zigzag<T>, 2, 4)

#define ZIGZAG_DEF(T,N) \
  def("zigzag", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,1>&, int, const bool))&Torch::ip::zigzag<T>, zigzag_overloads_ ## N ((arg("src"), arg("dst"), arg("n_coef")=0, arg("right_first")=false), ZIGZAG2D_DOC)); 

/*
ZIGZAG_DECL(bool,bool)
ZIGZAG_DECL(int8_t,int8)
ZIGZAG_DECL(int16_t,int16)
ZIGZAG_DECL(int32_t,int32)
ZIGZAG_DECL(int64_t,int64)
*/
ZIGZAG_DECL(uint8_t,uint8)
ZIGZAG_DECL(uint16_t,uint16)
/*
ZIGZAG_DECL(uint32_t,uint32)
ZIGZAG_DECL(uint64_t,uint64)
ZIGZAG_DECL(float,float32)
*/
ZIGZAG_DECL(double,float64)
/*
ZIGZAG_DECL(std::complex<float>,complex64)
ZIGZAG_DECL(std::complex<double>,complex128)
*/


void bind_ip_zigzag()
{
/*
  ZIGZAG_DEF(bool,bool)
  ZIGZAG_DEF(int8_t,int8)
  ZIGZAG_DEF(int16_t,int16)
  ZIGZAG_DEF(int32_t,int32)
  ZIGZAG_DEF(int64_t,int64)
*/
  ZIGZAG_DEF(uint8_t,uint8)
  ZIGZAG_DEF(uint16_t,uint16)
/*
  ZIGZAG_DEF(uint32_t,uint32)
  ZIGZAG_DEF(uint64_t,uint64)
  ZIGZAG_DEF(float,float32)
*/
  ZIGZAG_DEF(double,float64)
/*
  ZIGZAG_DEF(std::complex<float>,complex64)
  ZIGZAG_DEF(std::complex<double>,complex128)
*/
}
