/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 21 Feb 13:54:28 2011 
 *
 * @brief Implementation of MatUtils (handling of matlab .mat files)
 */

#include "database/MatUtils.h"

namespace db = Torch::database;
namespace array = Torch::core::array;
namespace det = db::detail;

enum matio_classes det::mio_class_type (array::ElementType i) {
  switch (i) {
    case array::t_int8: 
      return MAT_C_INT8;
    case array::t_int16: 
      return MAT_C_INT16;
    case array::t_int32: 
      return MAT_C_INT32;
    case array::t_int64: 
      return MAT_C_INT64;
    case array::t_uint8: 
      return MAT_C_UINT8;
    case array::t_uint16: 
      return MAT_C_UINT16;
    case array::t_uint32: 
      return MAT_C_UINT32;
    case array::t_uint64: 
      return MAT_C_UINT64;
    case array::t_float32:
      return MAT_C_SINGLE;
    case array::t_complex64:
      return MAT_C_SINGLE;
    case array::t_float64:
      return MAT_C_DOUBLE;
    case array::t_complex128:
      return MAT_C_DOUBLE;
    default:
      throw db::TypeError(i, array::t_float32);
  }
}

enum matio_types det::mio_data_type (array::ElementType i) {
  switch (i) {
    case array::t_int8: 
      return MAT_T_INT8;
    case array::t_int16: 
      return MAT_T_INT16;
    case array::t_int32: 
      return MAT_T_INT32;
    case array::t_int64: 
      return MAT_T_INT64;
    case array::t_uint8: 
      return MAT_T_UINT8;
    case array::t_uint16: 
      return MAT_T_UINT16;
    case array::t_uint32: 
      return MAT_T_UINT32;
    case array::t_uint64: 
      return MAT_T_UINT64;
    case array::t_float32:
      return MAT_T_SINGLE;
    case array::t_complex64:
      return MAT_T_SINGLE;
    case array::t_float64:
      return MAT_T_DOUBLE;
    case array::t_complex128:
      return MAT_T_DOUBLE;
    default:
      throw db::TypeError(i, array::t_float32);
  }
}

array::ElementType det::torch_element_type (int mio_type, bool is_complex) {

  array::ElementType eltype = array::t_unknown;

  switch(mio_type) {

    case(MAT_T_INT8): 
      eltype = array::t_int8;
      break;
    case(MAT_T_INT16): 
      eltype = array::t_int16;
      break;
    case(MAT_T_INT32):
      eltype = array::t_int32;
      break;
    case(MAT_T_INT64):
      eltype = array::t_int64;
      break;
    case(MAT_T_UINT8):
      eltype = array::t_uint8;
      break;
    case(MAT_T_UINT16):
      eltype = array::t_uint16;
      break;
    case(MAT_T_UINT32):
      eltype = array::t_uint32;
      break;
    case(MAT_T_UINT64):
      eltype = array::t_uint64;
      break;
    case(MAT_T_SINGLE):
      eltype = array::t_float32;
      break;
    case(MAT_T_DOUBLE):
      eltype = array::t_float64;
      break;
    default:
      return array::t_unknown;
  }

  //if type is complex, it is signalled slightly different
  if (is_complex) {
    if (eltype == array::t_float32) return array::t_complex64;
    else if (eltype == array::t_float64) return array::t_complex128;
    else return array::t_unknown;
  }
  
  return eltype;
}
  
template <> blitz::TinyVector<int,1> det::make_shape<1> (const int* shape) {
  return blitz::shape(shape[0]);
}

template <> blitz::TinyVector<int,2> det::make_shape<2> (const int* shape) {
  return blitz::shape(shape[0], shape[1]);
}

template <> blitz::TinyVector<int,3> det::make_shape<3> (const int* shape) {
  return blitz::shape(shape[0], shape[1], shape[2]);
}

template <> blitz::TinyVector<int,4> det::make_shape<4> (const int* shape) {
  return blitz::shape(shape[0], shape[1], shape[2], shape[3]);
}
