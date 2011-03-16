/**
 * @author <a href="mailto:laurent.el-shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Python bindings for Torch::core::convert. 
 *    Types supported are uint8, uint16 and float64
 */

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/preprocessor/stringize.hpp>

#include "core/convert.h"

using namespace boost::python;

static const char* ARRAY_CONVERT_DOC = "Return a blitz array of the specified type by converting the given blitz array.";
static const char* ARRAY_CONVERT_DOC_RANGE = "Return a blitz array of the specified type by converting the given blitz array, using the specified input and output ranges.";
static const char* ARRAY_CONVERT_DOC_TORANGE = "Return a blitz array of the specified type by converting the given blitz array, using the specified output range.";
static const char* ARRAY_CONVERT_DOC_FROMRANGE = "Return a blitz array of the specified type by converting the given blitz array, using the specified input range.";
#define ARRAY_CONVERT_DEF(T,NT,U,NU,D) \
  def(BOOST_PP_STRINGIZE(convert_ ## NU ## _ ## D), (blitz::Array<U,D> (*)(const blitz::Array<T,D>&))&Torch::core::convert<U,T,D>, (arg("array")), ARRAY_CONVERT_DOC); \
  def(BOOST_PP_STRINGIZE(convert_ ## NU ## _ ## D), (blitz::Array<U,D> (*)(const blitz::Array<T,D>&, U dst_min, U dst_max, T src_min, T src_max))&Torch::core::convert<U,T>, (arg("array"), arg("dst_min"), arg("dst_max"), arg("src_min"), arg("src_max")), ARRAY_CONVERT_DOC_RANGE); \
  def(BOOST_PP_STRINGIZE(convert_to_range_ ## NU ## _ ## D), (blitz::Array<U,D> (*)(const blitz::Array<T,D>&, U dst_min, U dst_max))&Torch::core::convertToRange<U,T,D>, (arg("array"), arg("dst_min"), arg("dst_max")), ARRAY_CONVERT_DOC_TORANGE); \
  def(BOOST_PP_STRINGIZE(convert_from_range_ ## NU ## _ ## D), (blitz::Array<U,D> (*)(const blitz::Array<T,D>&, T src_min, T src_max))&Torch::core::convertFromRange<U,T,D>, (arg("array"), arg("src_min"), arg("src_max")), ARRAY_CONVERT_DOC_FROMRANGE);

#define ARRAY_CONVERT_DEFS(U,N,D)\
  ARRAY_CONVERT_DEF(uint8_t,uint8,U,N,D) \
  ARRAY_CONVERT_DEF(uint16_t,uint16,U,N,D) \
  ARRAY_CONVERT_DEF(double,float64,U,N,D) 

void bind_core_convert() {
    ARRAY_CONVERT_DEFS(uint8_t, uint8, 1)
    ARRAY_CONVERT_DEFS(uint16_t, uint16, 1)
    ARRAY_CONVERT_DEFS(double, float64, 1)
}
