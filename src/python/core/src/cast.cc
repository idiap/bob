/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Python bindings for torch::core::cast
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "core/cast.h"

using namespace boost::python;

static const char* ARRAY_CAST_DOC = "Return a blitz array of the specified type by casting the given blitz array.";
#define ARRAY_CAST_DEF(T,NT,U,NU,D) def(BOOST_PP_STRINGIZE(cast_ ## NU ## _ ## D), (blitz::Array<U,D> (*)(const blitz::Array<T,D>&))&Torch::core::cast<U,T>, (arg("array")), ARRAY_CAST_DOC);

#define ARRAY_CAST_DEFS(U,N,D)\
  ARRAY_CAST_DEF(bool,bool,U,N,D) \
  ARRAY_CAST_DEF(int8_t,int8,U,N,D) \
  ARRAY_CAST_DEF(int16_t,int16,U,N,D) \
  ARRAY_CAST_DEF(int32_t,int32,U,N,D) \
  ARRAY_CAST_DEF(int64_t,int64,U,N,D) \
  ARRAY_CAST_DEF(uint8_t,uint8,U,N,D) \
  ARRAY_CAST_DEF(uint16_t,uint16,U,N,D) \
  ARRAY_CAST_DEF(uint32_t,uint32,U,N,D) \
  ARRAY_CAST_DEF(uint64_t,uint64,U,N,D) \
  ARRAY_CAST_DEF(float,float32,U,N,D) \
  ARRAY_CAST_DEF(double,float64,U,N,D) \
  ARRAY_CAST_DEF(long double,float128,U,N,D) \
  ARRAY_CAST_DEF(std::complex<float>,complex64,U,N,D) \
  ARRAY_CAST_DEF(std::complex<double>,complex128,U,N,D) \
  ARRAY_CAST_DEF(std::complex<long double>,complex256,U,N,D) 


void bind_core_cast() {
    ARRAY_CAST_DEFS(bool, bool, 1)
    ARRAY_CAST_DEFS(int8_t, int8, 1)
    ARRAY_CAST_DEFS(int16_t, int16, 1)
    ARRAY_CAST_DEFS(int32_t, int32, 1)
    ARRAY_CAST_DEFS(int64_t, int64, 1)
    ARRAY_CAST_DEFS(uint8_t, uint8, 1)
    ARRAY_CAST_DEFS(uint16_t, uint16, 1)
    ARRAY_CAST_DEFS(uint32_t, uint32, 1)
    ARRAY_CAST_DEFS(uint64_t, uint64, 1)
    ARRAY_CAST_DEFS(float, float32, 1)
    ARRAY_CAST_DEFS(double, float64, 1)
    ARRAY_CAST_DEFS(long double, float128, 1)
    ARRAY_CAST_DEFS(std::complex<float>, complex64, 1)
    ARRAY_CAST_DEFS(std::complex<double>, complex128, 1)
    ARRAY_CAST_DEFS(std::complex<long double>, complex256, 1)
    ARRAY_CAST_DEFS(bool, bool, 2)
    ARRAY_CAST_DEFS(int8_t, int8, 2)
    ARRAY_CAST_DEFS(int16_t, int16, 2)
    ARRAY_CAST_DEFS(int32_t, int32, 2)
    ARRAY_CAST_DEFS(int64_t, int64, 2)
    ARRAY_CAST_DEFS(uint8_t, uint8, 2)
    ARRAY_CAST_DEFS(uint16_t, uint16, 2)
    ARRAY_CAST_DEFS(uint32_t, uint32, 2)
    ARRAY_CAST_DEFS(uint64_t, uint64, 2)
    ARRAY_CAST_DEFS(float, float32, 2)
    ARRAY_CAST_DEFS(double, float64, 2)
    ARRAY_CAST_DEFS(long double, float128, 2)
    ARRAY_CAST_DEFS(std::complex<float>, complex64, 2)
    ARRAY_CAST_DEFS(std::complex<double>, complex128, 2)
    ARRAY_CAST_DEFS(std::complex<long double>, complex256, 2)
    ARRAY_CAST_DEFS(bool, bool, 3)
    ARRAY_CAST_DEFS(int8_t, int8, 3)
    ARRAY_CAST_DEFS(int16_t, int16, 3)
    ARRAY_CAST_DEFS(int32_t, int32, 3)
    ARRAY_CAST_DEFS(int64_t, int64, 3)
    ARRAY_CAST_DEFS(uint8_t, uint8, 3)
    ARRAY_CAST_DEFS(uint16_t, uint16, 3)
    ARRAY_CAST_DEFS(uint32_t, uint32, 3)
    ARRAY_CAST_DEFS(uint64_t, uint64, 3)
    ARRAY_CAST_DEFS(float, float32, 3)
    ARRAY_CAST_DEFS(double, float64, 3)
    ARRAY_CAST_DEFS(long double, float128, 3)
    ARRAY_CAST_DEFS(std::complex<float>, complex64, 3)
    ARRAY_CAST_DEFS(std::complex<double>, complex128, 3)
    ARRAY_CAST_DEFS(std::complex<long double>, complex256, 3)
    ARRAY_CAST_DEFS(bool, bool, 4)
    ARRAY_CAST_DEFS(int8_t, int8, 4)
    ARRAY_CAST_DEFS(int16_t, int16, 4)
    ARRAY_CAST_DEFS(int32_t, int32, 4)
    ARRAY_CAST_DEFS(int64_t, int64, 4)
    ARRAY_CAST_DEFS(uint8_t, uint8, 4)
    ARRAY_CAST_DEFS(uint16_t, uint16, 4)
    ARRAY_CAST_DEFS(uint32_t, uint32, 4)
    ARRAY_CAST_DEFS(uint64_t, uint64, 4)
    ARRAY_CAST_DEFS(float, float32, 4)
    ARRAY_CAST_DEFS(double, float64, 4)
    ARRAY_CAST_DEFS(long double, float128, 4)
    ARRAY_CAST_DEFS(std::complex<float>, complex64, 4)
    ARRAY_CAST_DEFS(std::complex<double>, complex128, 4)
    ARRAY_CAST_DEFS(std::complex<long double>, complex256, 4)
}
