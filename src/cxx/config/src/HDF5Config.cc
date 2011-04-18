/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Sat  5 Mar 20:26:56 2011 
 *
 * @brief Implementation of the Configuration main class
 */

#include <list>

#include "database/HDF5File.h"
#include "database/Arrayset.h"
#include "database/Array.h"

#include "config/HDF5Config.h"
#include "config/Exception.h"

namespace conf = Torch::config;
namespace db = Torch::database;
namespace bp = boost::python;

/**
 * Reads a list of scalars from the file and push it back into the given
 * dictionary.
 */
template <typename T> static 
void scalar_list_readout(db::HDF5File& f, const std::string& path,
    size_t length, bp::dict& dict) {
  std::list<T> list;
  for (size_t i=0; i<length; ++i) {
    T obj;
    f.read(path, i, obj);
    list.push_back(obj);
  }
  dict[path] = list;
}

/**
 * Reads a whole set of arrays in a HDF5 variable and push it back into the
 * given dictionary as a db::Arrayset
 */
template <typename T> static 
void single_arrayset_readout(db::HDF5File& f, const std::string& path,
    size_t length, bp::dict& dict) {
  db::Arrayset set;
  for (size_t i=0; i<length; ++i) {
    T obj;
    f.readArray(path, i, obj);
    set.add(db::Array(obj));
  }
  dict[path] = set;
}

/**
 * Reads a whole set of arrays with any (supported) number of dimensions and
 * push that back into the given dictionary as a db::Arrayset
 */
template <typename T> static
void arrayset_readout(db::HDF5File& f, const std::string& path, 
    size_t N, size_t length, bp::dict& dict) {
  switch (N) {
    case 1:
      single_arrayset_readout<blitz::Array<T,1> >(f, path, length, dict);
      break;
    case 2:
      single_arrayset_readout<blitz::Array<T,2> >(f, path, length, dict);
      break;
    case 3:
      single_arrayset_readout<blitz::Array<T,3> >(f, path, length, dict);
      break;
    case 4:
      single_arrayset_readout<blitz::Array<T,4> >(f, path, length, dict);
      break;
    default:
      throw db::HDF5UnsupportedDimensionError(N);
      break;
  }
}

/**
 * Main routine for reading a list of scalars from the file and placing it into
 * the given dictionary
 */
static void main_scalar_list_readout(db::HDF5File& f, const std::string& path,
    db::hdf5type type, size_t length, bp::dict& dict) {
  switch (type) {
    case db::i8:
      scalar_list_readout<int8_t>(f, path, length, dict);
      break;
    case db::i16:
      scalar_list_readout<int16_t>(f, path, length, dict);
      break;
    case db::i32:
      scalar_list_readout<int32_t>(f, path, length, dict);
      break;
    case db::i64:
      scalar_list_readout<int64_t>(f, path, length, dict);
      break;
    case db::u8:
      scalar_list_readout<uint8_t>(f, path, length, dict);
      break;
    case db::u16:
      scalar_list_readout<uint16_t>(f, path, length, dict);
      break;
    case db::u32:
      scalar_list_readout<uint32_t>(f, path, length, dict);
      break;
    case db::u64:
      scalar_list_readout<uint64_t>(f, path, length, dict);
      break;
    case db::f32:
      scalar_list_readout<float>(f, path, length, dict);
      break;
    case db::f64:
      scalar_list_readout<double>(f, path, length, dict);
      break;
    case db::f128:
      scalar_list_readout<long double>(f, path, length, dict);
      break;
    case db::c64:
      scalar_list_readout<std::complex<float> >(f, path, length, dict);
      break;
    case db::c128:
      scalar_list_readout<std::complex<double> >(f, path, length, dict);
      break;
    case db::c256:
      scalar_list_readout<std::complex<long double> >(f, path, length, dict);
      break;
    default:
      throw db::HDF5UnsupportedTypeError();
      break;
  }
}

/**
 * Main routine to read a list of blitz::Array<>'s from the file and place the
 * as a db::Arrayset into the given dictionary.
 */
static void main_arrayset_readout(db::HDF5File& f, const std::string& path, 
    db::hdf5type type, size_t ndims, size_t length, bp::dict& dict) {
  switch (type) {
    case db::i8:
      arrayset_readout<int8_t>(f, path, ndims, length, dict);
      break;
    case db::i16:
      arrayset_readout<int16_t>(f, path, ndims, length, dict);
      break;
    case db::i32:
      arrayset_readout<int32_t>(f, path, ndims, length, dict);
      break;
    case db::i64:
      arrayset_readout<int64_t>(f, path, ndims, length, dict);
      break;
    case db::u8:
      arrayset_readout<uint8_t>(f, path, ndims, length, dict);
      break;
    case db::u16:
      arrayset_readout<uint16_t>(f, path, ndims, length, dict);
      break;
    case db::u32:
      arrayset_readout<uint32_t>(f, path, ndims, length, dict);
      break;
    case db::u64:
      arrayset_readout<uint64_t>(f, path, ndims, length, dict);
      break;
    case db::f32:
      arrayset_readout<float>(f, path, ndims, length, dict);
      break;
    case db::f64:
      arrayset_readout<double>(f, path, ndims, length, dict);
      break;
    case db::f128:
      arrayset_readout<long double>(f, path, ndims, length, dict);
      break;
    case db::c64:
      arrayset_readout<std::complex<float> >(f, path, ndims, length, dict);
      break;
    case db::c128:
      arrayset_readout<std::complex<double> >(f, path, ndims, length, dict);
      break;
    case db::c256:
      arrayset_readout<std::complex<long double> >(f, path, ndims, length, dict);
      break;
    default:
      throw db::HDF5UnsupportedTypeError();
      break;
  }
}

void conf::detail::hdf5load(const boost::filesystem::path& path,
    bp::dict& dict) {
  db::HDF5File f(path.string(), db::HDF5File::in);
  std::vector<std::string> variables;
  f.paths(variables);
  for (size_t i=0; i<variables.size(); ++i) {
    const db::HDF5Type& descr = f.describe(variables[i]);
    size_t length = f.size(variables[i]);
    if (length) {
      if (!descr.shape()) main_scalar_list_readout(f, variables[i], descr.type(), length, dict);
      else main_arrayset_readout(f, variables[i], descr.type(), descr.shape().n(), length, dict);
    }
  }
}

template <typename T> static void add_array(db::HDF5File& f,
    const std::string& path, const db::Array array) {
  switch (array.getNDim()) {
    case 1:
      f.appendArray(path, array.get<T,1>());
      break;
    case 2:
      f.appendArray(path, array.get<T,2>());
      break;
    case 3:
      f.appendArray(path, array.get<T,3>());
      break;
    case 4:
      f.appendArray(path, array.get<T,4>());
      break;
    default:
      throw db::HDF5UnsupportedDimensionError(array.getNDim());
  }
}

static bool save_as_arrayset(db::HDF5File& f, const std::string& path,
    bp::object obj) {
  bp::extract<db::Arrayset> extor(obj);
  if (!extor.check()) return false;
  db::Arrayset set = extor();
  std::vector<size_t> ids;
  set.index(ids);
  for (size_t i=0; i<ids.size(); ++i) {
    switch(set.getElementType()) {
      case Torch::core::array::t_bool:
        add_array<bool>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_int8:
        add_array<int8_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_int16:
        add_array<int16_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_int32:
        add_array<int32_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_int64:
        add_array<int64_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_uint8:
        add_array<uint8_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_uint16:
        add_array<uint16_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_uint32:
        add_array<uint32_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_uint64:
        add_array<uint64_t>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_float32:
        add_array<float>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_float64:
        add_array<double>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_float128:
        add_array<long double>(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_complex64:
        add_array<std::complex<float> >(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_complex128:
        add_array<std::complex<double> >(f, path, set[ids[i]]);
        break;
      case Torch::core::array::t_complex256:
        add_array<std::complex<long double> >(f, path, set[ids[i]]);
        break;
      default:
        throw db::HDF5UnsupportedTypeError();
        break;
    }
  }
  return true;
}

/**
 * To save an array is a special case of the arrayset. Convert the array into
 * an arrayset and call save as arrayset above.
 */
static bool save_as_array(db::HDF5File& f, const std::string& path,
    bp::object obj) {
  bp::extract<db::Array> extor(obj);
  if (!extor.check()) return false;
  db::Arrayset tmp;
  tmp.add(extor());
  return save_as_arrayset(f, path, bp::object(tmp));
}

/**
 * Try saving with a particular type of list
 */
template <typename T>
static bool try_t_list(db::HDF5File& f,
    const std::string& path, bp::object obj) {
  typedef typename std::list<T> tlist;
  typedef typename std::list<T>::const_iterator itype;
  bp::extract<const tlist&> extor(obj);
  if (!extor.check()) return false;
  const tlist& l = extor();
  for (itype it=l.begin(); it!=l.end(); ++it) f.append(path, *it);
  return true;
}

/**
 * Saving as a scalar list
 */
static bool save_as_scalar_list(db::HDF5File& f, const std::string& path,
    bp::object obj) {
  //if (try_t_list<std::string>(f, path, obj)) return;
  if (try_t_list<bool>(f, path, obj)) return true;
  if (try_t_list<int8_t>(f, path, obj)) return true;
  if (try_t_list<int16_t>(f, path, obj)) return true;
  if (try_t_list<int32_t>(f, path, obj)) return true;
  if (try_t_list<int64_t>(f, path, obj)) return true;
  if (try_t_list<uint8_t>(f, path, obj)) return true;
  if (try_t_list<uint16_t>(f, path, obj)) return true;
  if (try_t_list<uint32_t>(f, path, obj)) return true;
  if (try_t_list<uint64_t>(f, path, obj)) return true;
  if (try_t_list<float>(f, path, obj)) return true;
  if (try_t_list<double>(f, path, obj)) return true;
  if (try_t_list<long double>(f, path, obj)) return true;
  if (try_t_list<std::complex<float> >(f, path, obj)) return true;
  if (try_t_list<std::complex<double> >(f, path, obj)) return true;
  if (try_t_list<std::complex<long double> >(f, path, obj)) return true;
  return false;
}

template <typename T>
static bool try_t(db::HDF5File& f, const std::string& path, bp::object obj) {
  bp::extract<T> extor(obj);
  if (!extor.check()) return false;
  std::list<T> tmp;
  tmp.push_back(extor());
  return save_as_scalar_list(f, path, bp::object(tmp));
}

/**
 * Saving a single scalar is a special case of the scalar_list. Please note
 * that single scalars will be treated in a slightly different way than C++
 * types in general as python only has ints, longs, floats (double-precision)
 * and complex128 (double-precision complex). So, if you want to save scalars
 * with smaller precision, we will have to cast.
 */
static bool save_as_scalar(db::HDF5File& f, const std::string& path,
    bp::object obj) {
  //if (try_t<std::string>(f, path, obj)) return true;
  if (try_t<int64_t>(f, path, obj)) return true;
  if (try_t<double>(f, path, obj)) return true;
  if (try_t<std::complex<double> >(f, path, obj)) return true;
  return false;
}

/**
 * Saving requires that objects in dict are one of the following:
 * a. A scalar of any supported type
 * b. A list of scalars
 * c. A db::Array
 * d. A db::Arrayset
 */
void conf::detail::hdf5save(const boost::filesystem::path& path,
    const bp::dict& dict) {
  bp::list keys = dict.keys();
  db::HDF5File f(path.string(), db::HDF5File::trunc);
  for (Py_ssize_t i=0; i<bp::len(keys); ++i) {
    const char* varname = bp::extract<const char*>(keys[i]);
    if (save_as_arrayset(f, varname, dict[keys[i]])) continue;
    if (save_as_array(f, varname, dict[keys[i]])) continue;
    if (save_as_scalar_list(f, varname, dict[keys[i]])) continue;
    if (save_as_scalar(f, varname, dict[keys[i]])) continue;
    throw conf::UnsupportedConversion(varname, typeid(db::Arrayset), dict[keys[i]]);
  }
}
