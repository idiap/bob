/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Python bindings for Torch::core::is{Zero,One}Base, etc. 
 *    Types supported are uint8, uint16 and float64
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "core/array_check.h"

using namespace boost::python;

static const char* ZERO_BASE_DOC = "Checks that a blitz array has zero base indices.";
static const char* ONE_BASE_DOC = "Checks that a blitz array has one base indices.";
static const char* C_CONTIGUOUS_DOC = "Checks that a blitz array is stored contiguously in memory, in row-major order (C-style).";
static const char* FORTRAN_CONTIGUOUS_DOC = "Checks that a blitz array is stored contiguously in memory, in column-major order (Fortran-like).";
static const char* C_ZERO_BASE_CONTIGUOUS_DOC = "Checks that a blitz array is stored contiguously in memory, in row-major order (C-style), and with zero base indices.";
static const char* FORTRAN_ONE_BASE_CONTIGUOUS_DOC = "Checks that a blitz array is stored contiguously in memory, in column-major order (Fortran-like), and with one base indices.";
#define ARRAY_CHECK_DEF(T,D) \
  def("isZeroBase", (bool (*)(const blitz::Array<T,D>&))&Torch::core::array::isZeroBase<T,D>, (arg("array")), ZERO_BASE_DOC); \
  def("isOneBase", (bool (*)(const blitz::Array<T,D>&))&Torch::core::array::isOneBase<T,D>, (arg("array")), ONE_BASE_DOC); \
  def("isCContiguous", (bool (*)(const blitz::Array<T,D>&))&Torch::core::array::isCContiguous<T,D>, (arg("array")), C_CONTIGUOUS_DOC); \
  def("isFortranContiguous", (bool (*)(const blitz::Array<T,D>&))&Torch::core::array::isFortranContiguous<T,D>, (arg("array")), FORTRAN_CONTIGUOUS_DOC); \
  def("isCZeroBaseContiguous", (bool (*)(const blitz::Array<T,D>&))&Torch::core::array::isCZeroBaseContiguous<T,D>, (arg("array")), C_ZERO_BASE_CONTIGUOUS_DOC); \
  def("isFortranOneBaseContiguous", (bool (*)(const blitz::Array<T,D>&))&Torch::core::array::isFortranOneBaseContiguous<T,D>, (arg("array")), FORTRAN_ONE_BASE_CONTIGUOUS_DOC); 

#define ARRAY_CHECK_DEFS(T)\
  ARRAY_CHECK_DEF(T,1) \
  ARRAY_CHECK_DEF(T,2) \
  ARRAY_CHECK_DEF(T,3) \
  ARRAY_CHECK_DEF(T,4) 

void bind_core_array_check() {
    ARRAY_CHECK_DEFS(bool)
    ARRAY_CHECK_DEFS(int8_t)
    ARRAY_CHECK_DEFS(int16_t)
    ARRAY_CHECK_DEFS(int32_t)
    ARRAY_CHECK_DEFS(int64_t)
    ARRAY_CHECK_DEFS(uint8_t)
    ARRAY_CHECK_DEFS(uint16_t)
    ARRAY_CHECK_DEFS(uint32_t)
    ARRAY_CHECK_DEFS(uint64_t)
    ARRAY_CHECK_DEFS(float)
    ARRAY_CHECK_DEFS(double)
    ARRAY_CHECK_DEFS(long double)
    ARRAY_CHECK_DEFS(std::complex<float>)
    ARRAY_CHECK_DEFS(std::complex<double>)
    ARRAY_CHECK_DEFS(std::complex<long double>)
}
