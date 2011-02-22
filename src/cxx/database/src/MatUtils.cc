/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 21 Feb 13:54:28 2011 
 *
 * @brief Implementation of MatUtils (handling of matlab .mat files)
 */

#include "database/MatUtils.h"
#include <boost/regex.hpp>
#include <boost/lexical_cast.hpp>

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

void det::get_info(const matvar_t* matvar,
    Torch::database::ArrayTypeInfo& info) {
  info.ndim = matvar->rank;
  for (size_t i=0; i<info.ndim; ++i) info.shape[i] = matvar->dims[i];
  info.eltype = db::detail::torch_element_type(matvar->data_type,
      matvar->isComplex);
}

void det::get_info_first(const std::string& filename, db::ArrayTypeInfo& info) {

  static const boost::regex allowed_varname("^array_(\\d*)$");
  boost::cmatch what;

  boost::shared_ptr<std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> > > retval(new std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> >());

  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
  
  if (!mat) throw db::FileNotReadable(filename);
  
  matvar_t* matvar = Mat_VarReadNext(mat); //gets the first variable

  //we continue reading until we find a variable that matches our naming
  //convention.
  while (matvar && !boost::regex_match(matvar->name, what, allowed_varname)) {
    Mat_VarFree(matvar);
    matvar = Mat_VarReadNext(mat); //gets the first variable
  }

  if (!what.size()) {
    Mat_Close(mat);
    throw db::Uninitialized();
  }

  get_info(matvar, info);

  Mat_VarFree(matvar);
  Mat_Close(mat);
}

boost::shared_ptr<std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> > >
  det::list_variables(const std::string& filename) {

  static const boost::regex allowed_varname("^array_(\\d*)$");
  boost::cmatch what;

  boost::shared_ptr<std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> > > retval(new std::map<size_t, std::pair<std::string, db::ArrayTypeInfo> >());

  mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
  if (!mat) throw db::FileNotReadable(filename);
  matvar_t* matvar = Mat_VarReadNext(mat); //gets the first variable

  //we continue reading until we find a variable that matches our naming
  //convention.
  while (matvar && !boost::regex_match(matvar->name, what, allowed_varname)) {
    Mat_VarFree(matvar);
    matvar = Mat_VarReadNext(mat); //gets the first variable
  }

  if (!what.size()) throw db::Uninitialized();

  size_t id = boost::lexical_cast<size_t>(what[1]);
 
  //now that we have found a variable under our name convention, fill the array
  //properties taking that variable as basis
  (*retval)[id] = std::make_pair(matvar->name, db::ArrayTypeInfo());
  get_info(matvar, (*retval)[id].second);

  //release this one and go for reading the next ones
  Mat_VarFree(matvar);

  //checks our support and see if we can load this...
  if ((*retval)[id].second.ndim > 4) {
    Mat_Close(mat);
    throw db::DimensionError((*retval)[id].second.ndim, 
        Torch::core::array::N_MAX_DIMENSIONS_ARRAY);
  }
  if ((*retval)[id].second.eltype == Torch::core::array::t_unknown) {
    Mat_Close(mat);
    throw db::TypeError((*retval)[id].second.eltype, 
        Torch::core::array::t_float32);
  }

  //if we got here, just continue counting the variables inside. we
  //only read their info since that is faster

  while ((matvar = Mat_VarReadNextInfo(mat))) {
    if (boost::regex_match(matvar->name, what, allowed_varname)) {
      id = boost::lexical_cast<size_t>(what[1]);
      (*retval)[id] = std::make_pair(matvar->name, db::ArrayTypeInfo());
      get_info(matvar, (*retval)[id].second);
    }
    Mat_VarFree(matvar);
  }

  Mat_Close(mat);

  return retval;
}
