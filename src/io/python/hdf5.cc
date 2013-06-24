/**
 * @file io/python/hdf5.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Binds our C++ HDF5 interface to python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>

#include <bob/python/exception.h>
#include <bob/python/ndarray.h>

#include <bob/io/HDF5File.h>

using namespace boost::python;

/**
 * Allows us to write HDF5File("filename.hdf5", "r")
 */
static boost::shared_ptr<bob::io::HDF5File>
hdf5file_make_fromstr(const std::string& filename, const std::string& opmode) {
  if (opmode.size() > 1) PYTHON_ERROR(RuntimeError, "Supported flags are 'r' (read-only), 'a' (read/write/append), 'w' (read/write/truncate) or 'x' (read/write/exclusive), but you tried to use '%s'", opmode.c_str());
  bob::io::HDF5File::mode_t mode = bob::io::HDF5File::inout;
  if (opmode[0] == 'r') mode = bob::io::HDF5File::in;
  else if (opmode[0] == 'a') mode = bob::io::HDF5File::inout;
  else if (opmode[0] == 'w') mode = bob::io::HDF5File::trunc;
  else if (opmode[0] == 'x') mode = bob::io::HDF5File::excl;
  else { //anything else is just unsupported for the time being
    PYTHON_ERROR(RuntimeError, "Supported flags are 'r' (read-only), 'a' (read/write/append), 'w' (read/write/truncate) or 'x' (read/write/exclusive), but you tried to use '%s'", opmode.c_str());
  }
  return boost::make_shared<bob::io::HDF5File>(filename, mode);
}

/**
 * Returns a list of all paths inside a HDF5File
 */
static list hdf5file_paths(const bob::io::HDF5File& f, const bool relative) {
  list retval;
  std::vector<std::string> values;
  f.paths(values, relative);
  for (size_t i=0; i<values.size(); ++i) retval.append(str(values[i]));
  return retval;
}

/**
 * Returns a list of all sub-directories inside a HDF5File
 */
static list hdf5file_sub_groups(const bob::io::HDF5File& f, const bool relative, bool recursive) {
  list retval;
  std::vector<std::string> values;
  f.sub_groups(values, relative, recursive);
  for (size_t i=0; i<values.size(); ++i) retval.append(str(values[i]));
  return retval;
}

/**
 * Returns tuples for the description of all possible ways to read a certain
 * path.
 */
static tuple hdf5file_describe(const bob::io::HDF5File& f, const std::string& p) {
  const std::vector<bob::io::HDF5Descriptor>& dv = f.describe(p);
  list retval;
  for (size_t k=0; k<dv.size(); ++k) retval.append(dv[k]);
  return tuple(retval);
}

/**
 * Functionality to read from HDF5File's
 */
static object hdf5file_xread(bob::io::HDF5File& f, const std::string& p,
    int descriptor, int pos) {

  const std::vector<bob::io::HDF5Descriptor>& D = f.describe(p);

  //last descriptor always contains the full readout.
  const bob::io::HDF5Type& type = D[descriptor].type;
  const bob::io::HDF5Shape& shape = type.shape();

  if (shape.n() == 1 && shape[0] == 1) { //read as scalar
    switch(type.type()) {
      case bob::io::s:
        return object(f.read<std::string>(p, pos));
      case bob::io::b:
        return object(f.read<bool>(p, pos));
      case bob::io::i8:
        return object(f.read<int8_t>(p, pos));
      case bob::io::i16:
        return object(f.read<int16_t>(p, pos));
      case bob::io::i32:
        return object(f.read<int32_t>(p, pos));
      case bob::io::i64:
        return object(f.read<int64_t>(p, pos));
      case bob::io::u8:
        return object(f.read<uint8_t>(p, pos));
      case bob::io::u16:
        return object(f.read<uint16_t>(p, pos));
      case bob::io::u32:
        return object(f.read<uint32_t>(p, pos));
      case bob::io::u64:
        return object(f.read<uint64_t>(p, pos));
      case bob::io::f32:
        return object(f.read<float>(p, pos));
      case bob::io::f64:
        return object(f.read<double>(p, pos));
      case bob::io::f128:
        return object(f.read<long double>(p, pos));
      case bob::io::c64:
        return object(f.read<std::complex<float> >(p, pos));
      case bob::io::c128:
        return object(f.read<std::complex<double> >(p, pos));
      case bob::io::c256:
        return object(f.read<std::complex<long double> >(p, pos));
      default:
        PYTHON_ERROR(TypeError, "unsupported HDF5 type: %s", type.str().c_str());
    }
  }

  //read as an numpy array
  bob::core::array::typeinfo atype;
  type.copy_to(atype);
  bob::python::py_array retval(atype);
  f.read_buffer(p, pos, atype, retval.ptr());
  return retval.pyobject();
}

static object hdf5file_lread(bob::io::HDF5File& f, const std::string& p,
    int64_t pos=-1) {
  if (pos >= 0) return hdf5file_xread(f, p, 0, pos);

  //otherwise returns as a list
  const std::vector<bob::io::HDF5Descriptor>& D = f.describe(p);
  list retval;
  for (uint64_t k=0; k<D[0].size; ++k)
    retval.append(hdf5file_xread(f, p, 0, k));
  return retval;
}
BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_lread_overloads, hdf5file_lread, 2, 3)

static inline object hdf5file_read(bob::io::HDF5File& f, const std::string& p) {
  return hdf5file_xread(f, p, 1, 0);
}

void set_string_type(bob::io::HDF5Type& t, object o) {
  t = bob::io::HDF5Type(extract<std::string>(o));
}

template <typename T> void set_type(bob::io::HDF5Type& t) {
  T v;
  t = bob::io::HDF5Type(v);
}

/**
 * Sets at 't', the type of the object 'o' according to our support types.
 * Raise in case of problems. Furthermore, returns 'true' if the object is as
 * simple scalar.
 */
static bool get_object_type(object o, bob::io::HDF5Type& t) {
  PyObject* op = o.ptr();

  if (PyArray_IsAnyScalar(op)) {
    if (PyArray_IsScalar(op, String)) set_string_type(t, o);
    else if (PyString_Check(op)) set_string_type(t, o);
    else if (PyBool_Check(op)) set_type<bool>(t);
    else if (PyInt_Check(op)) set_type<int32_t>(t);
    else if (PyLong_Check(op)) set_type<int64_t>(t);
    else if (PyFloat_Check(op)) set_type<double>(t);
    else if (PyComplex_Check(op)) set_type<std::complex<double> >(t);
    else if (PyArray_IsScalar(op, Bool)) set_type<bool>(t);
    else if (PyArray_IsScalar(op, Int8)) set_type<int8_t>(t);
    else if (PyArray_IsScalar(op, UInt8)) set_type<uint8_t>(t);
    else if (PyArray_IsScalar(op, Int16)) set_type<int16_t>(t);
    else if (PyArray_IsScalar(op, UInt16)) set_type<uint16_t>(t);
    else if (PyArray_IsScalar(op, Int32)) set_type<int32_t>(t);
    else if (PyArray_IsScalar(op, UInt32)) set_type<uint32_t>(t);
    else if (PyArray_IsScalar(op, Int64)) set_type<int64_t>(t);
    else if (PyArray_IsScalar(op, UInt64)) set_type<uint64_t>(t);
    else if (PyArray_IsScalar(op, Float)) set_type<float>(t);
    else if (PyArray_IsScalar(op, Double)) set_type<double>(t);
    else if (PyArray_IsScalar(op, LongDouble)) set_type<long double>(t);
    else if (PyArray_IsScalar(op, CFloat)) set_type<std::complex<float> >(t);
    else if (PyArray_IsScalar(op, CDouble)) set_type<std::complex<double> >(t);
    else if (PyArray_IsScalar(op, CLongDouble)) set_type<std::complex<long double> >(t);
    else {
      str so(o);
      std::string s = extract<std::string>(so);
      PYTHON_ERROR(TypeError, "No support for HDF5 type conversion for scalar object '%s'", s.c_str());
    }
    return true;
  }

  else if (PyArray_Check(op)) {
    bob::core::array::typeinfo ti;
    bob::python::typeinfo_ndarray_(o, ti);
    t = bob::io::HDF5Type(ti);
    return false;
  }

  else {
    //checks for convertibility to numpy.ndarray (not necessarily writeable,
    //but has to be "behaved" = C-style contiguous).
    bob::core::array::typeinfo ti;
    if (bob::python::convertible(o, ti, false, true) != bob::python::IMPOSSIBLE) {
      t = bob::io::HDF5Type(ti);
      return false;
    }
  }

  //if you get to this point, then this object is not supported
  str so(o);
  std::string printout = extract<std::string>(so);
  PYTHON_ERROR(TypeError, "No support for HDF5 type conversion for object of unknown type %s", printout.c_str());
}

template <typename T>
static void inner_replace_scalar(bob::io::HDF5File& f,
  const std::string& path, object obj, size_t pos) {
  T value = extract<T>(obj);
  f.replace(path, pos, value);
}

static void inner_replace(bob::io::HDF5File& f, const std::string& path,
    const bob::io::HDF5Type& type, object obj, size_t pos, bool scalar) {

  //no error detection: this should be done before reaching this method

  if (scalar) { //write as a scalar
    switch(type.type()) {
      case bob::io::s:  
        return inner_replace_scalar<std::string>(f, path, obj, pos);
      case bob::io::b:  
        return inner_replace_scalar<bool>(f, path, obj, pos);
      case bob::io::i8:  
        return inner_replace_scalar<int8_t>(f, path, obj, pos);
      case bob::io::i16: 
        return inner_replace_scalar<int16_t>(f, path, obj, pos);
      case bob::io::i32:
        return inner_replace_scalar<int32_t>(f, path, obj, pos);
      case bob::io::i64: 
        return inner_replace_scalar<int64_t>(f, path, obj, pos);
      case bob::io::u8:  
        return inner_replace_scalar<uint8_t>(f, path, obj, pos);
      case bob::io::u16: 
        return inner_replace_scalar<uint16_t>(f, path, obj, pos);
      case bob::io::u32: 
        return inner_replace_scalar<uint32_t>(f, path, obj, pos);
      case bob::io::u64: 
        return inner_replace_scalar<uint64_t>(f, path, obj, pos);
      case bob::io::f32: 
        return inner_replace_scalar<float>(f, path, obj, pos);
      case bob::io::f64: 
        return inner_replace_scalar<double>(f, path, obj, pos);
      case bob::io::f128: 
        return inner_replace_scalar<long double>(f, path, obj, pos);
      case bob::io::c64:
        return inner_replace_scalar<std::complex<float> >(f, path, obj, pos);
      case bob::io::c128:
        return inner_replace_scalar<std::complex<double> >(f, path, obj, pos);
      case bob::io::c256:
        return inner_replace_scalar<std::complex<long double> >(f, path, obj, pos);
      default:
        break;
    }
  }

  else { //write as an numpy array
    bob::python::py_array tmp(obj, object());
    f.write_buffer(path, pos, tmp.type(), tmp.ptr());
  }
}

static void hdf5file_replace(bob::io::HDF5File& f, const std::string& path,
    size_t pos, object obj) {
  bob::io::HDF5Type type;
  bool scalar = get_object_type(obj, type);
  inner_replace(f, path, type, obj, pos, scalar);
}

template <typename T>
static void inner_append_scalar(bob::io::HDF5File& f, const std::string& path,
    object obj) {
  T value = extract<T>(obj);
  f.append(path, value);
}

static void inner_append(bob::io::HDF5File& f, const std::string& path,
    const bob::io::HDF5Type& type, object obj, size_t compression, bool scalar) {

  //no error detection: this should be done before reaching this method

  if (scalar) { //write as a scalar
    switch(type.type()) {
      case bob::io::s:  
        return inner_append_scalar<std::string>(f, path, obj);
      case bob::io::b:  
        return inner_append_scalar<bool>(f, path, obj);
      case bob::io::i8:  
        return inner_append_scalar<int8_t>(f, path, obj);
      case bob::io::i16: 
        return inner_append_scalar<int16_t>(f, path, obj);
      case bob::io::i32:
        return inner_append_scalar<int32_t>(f, path, obj);
      case bob::io::i64: 
        return inner_append_scalar<int64_t>(f, path, obj);
      case bob::io::u8:  
        return inner_append_scalar<uint8_t>(f, path, obj);
      case bob::io::u16: 
        return inner_append_scalar<uint16_t>(f, path, obj);
      case bob::io::u32: 
        return inner_append_scalar<uint32_t>(f, path, obj);
      case bob::io::u64: 
        return inner_append_scalar<uint64_t>(f, path, obj);
      case bob::io::f32: 
        return inner_append_scalar<float>(f, path, obj);
      case bob::io::f64: 
        return inner_append_scalar<double>(f, path, obj);
      case bob::io::f128: 
        return inner_append_scalar<long double>(f, path, obj);
      case bob::io::c64:
        return inner_append_scalar<std::complex<float> >(f, path, obj);
      case bob::io::c128:
        return inner_append_scalar<std::complex<double> >(f, path, obj);
      case bob::io::c256:
        return inner_append_scalar<std::complex<long double> >(f, path, obj);
      default:
        break;
    }
  }

  else { //write as an numpy array
    bob::python::py_array tmp(obj, object());
    if (!f.contains(path)) f.create(path, tmp.type(), true, compression);
    f.extend_buffer(path, tmp.type(), tmp.ptr());
  }
}

static void hdf5file_append_iterable(bob::io::HDF5File& f, const std::string& path,
  object iterable, size_t compression) {
  for (int k=0; k<len(iterable); ++k) {
    object obj = iterable[k];
    bob::io::HDF5Type type;
    bool scalar = get_object_type(obj, type);
    inner_append(f, path, type, obj, compression, scalar);
  }
}

static void hdf5file_append(bob::io::HDF5File& f, const std::string& path,
    object obj, size_t compression=0) {
  PyObject* op = obj.ptr();
  if (PyList_Check(op) || PyTuple_Check(op)) {
    hdf5file_append_iterable(f, path, obj, compression);
  }
  else {
    bob::io::HDF5Type type;
    bool scalar = get_object_type(obj, type);
    inner_append(f, path, type, obj, compression, scalar);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_append_overloads, hdf5file_append, 3, 4)

template <typename T>
static void inner_set_scalar(bob::io::HDF5File& f, const std::string& path,
    object obj) {
  T value = extract<T>(obj);
  f.set(path, value);
}

static void inner_set(bob::io::HDF5File& f, const std::string& path,
    const bob::io::HDF5Type& type, object obj, size_t compression, bool scalar) {

  //no error detection: this should be done before reaching this method

  if (scalar) { //write as a scalar
    switch(type.type()) {
      case bob::io::s:  
        return inner_set_scalar<std::string>(f, path, obj);
      case bob::io::b:  
        return inner_set_scalar<bool>(f, path, obj);
      case bob::io::i8:  
        return inner_set_scalar<int8_t>(f, path, obj);
      case bob::io::i16: 
        return inner_set_scalar<int16_t>(f, path, obj);
      case bob::io::i32:
        return inner_set_scalar<int32_t>(f, path, obj);
      case bob::io::i64: 
        return inner_set_scalar<int64_t>(f, path, obj);
      case bob::io::u8:  
        return inner_set_scalar<uint8_t>(f, path, obj);
      case bob::io::u16: 
        return inner_set_scalar<uint16_t>(f, path, obj);
      case bob::io::u32: 
        return inner_set_scalar<uint32_t>(f, path, obj);
      case bob::io::u64: 
        return inner_set_scalar<uint64_t>(f, path, obj);
      case bob::io::f32: 
        return inner_set_scalar<float>(f, path, obj);
      case bob::io::f64: 
        return inner_set_scalar<double>(f, path, obj);
      case bob::io::f128: 
        return inner_set_scalar<long double>(f, path, obj);
      case bob::io::c64:
        return inner_set_scalar<std::complex<float> >(f, path, obj);
      case bob::io::c128:
        return inner_set_scalar<std::complex<double> >(f, path, obj);
      case bob::io::c256:
        return inner_set_scalar<std::complex<long double> >(f, path, obj);
      default:
        break;
    }
  }

  else { //write as an numpy array
    bob::python::py_array tmp(obj, object());
    if (!f.contains(path)) f.create(path, tmp.type(), false, compression);
    f.write_buffer(path, 0, tmp.type(), tmp.ptr());
  }
}

static void hdf5file_set(bob::io::HDF5File& f, const std::string& path,
    object obj, size_t compression=0) {
  bob::io::HDF5Type type;
  bool scalar = get_object_type(obj, type);
  inner_set(f, path, type, obj, compression, scalar);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_set_overloads, hdf5file_set, 3, 4)

template <typename T>
static object inner_get_scalar_attr(const bob::io::HDF5File& f,
  const std::string& path, const std::string& name, const bob::io::HDF5Type& type) {
  T value;
  f.read_attribute(path, name, type, static_cast<void*>(&value));
  return object(value);
}

template <>
object inner_get_scalar_attr<std::string>(const bob::io::HDF5File& f,
  const std::string& path, const std::string& name, const bob::io::HDF5Type&) {
  std::string retval;
  f.getAttribute(path, name, retval);
  return object(retval);
}

static object inner_get_attr(const bob::io::HDF5File& f, const std::string& path,
    const std::string& name, const bob::io::HDF5Type& type) {

  //no error detection: this should be done before reaching this method

  const bob::io::HDF5Shape& shape = type.shape();

  if (type.type() == bob::io::s || (shape.n() == 1 && shape[0] == 1)) { 
    //read as scalar
    switch(type.type()) {
      case bob::io::s:  
        return inner_get_scalar_attr<std::string>(f, path, name, type);
      case bob::io::b:  
        return inner_get_scalar_attr<bool>(f, path, name, type);
      case bob::io::i8:  
        return inner_get_scalar_attr<int8_t>(f, path, name, type);
      case bob::io::i16: 
        return inner_get_scalar_attr<int16_t>(f, path, name, type);
      case bob::io::i32:
        return inner_get_scalar_attr<int32_t>(f, path, name, type);
      case bob::io::i64: 
        return inner_get_scalar_attr<int64_t>(f, path, name, type);
      case bob::io::u8:  
        return inner_get_scalar_attr<uint8_t>(f, path, name, type);
      case bob::io::u16: 
        return inner_get_scalar_attr<uint16_t>(f, path, name, type);
      case bob::io::u32: 
        return inner_get_scalar_attr<uint32_t>(f, path, name, type);
      case bob::io::u64: 
        return inner_get_scalar_attr<uint64_t>(f, path, name, type);
      case bob::io::f32: 
        return inner_get_scalar_attr<float>(f, path, name, type);
      case bob::io::f64: 
        return inner_get_scalar_attr<double>(f, path, name, type);
      case bob::io::f128: 
        return inner_get_scalar_attr<long double>(f, path, name, type);
      case bob::io::c64:
        return inner_get_scalar_attr<std::complex<float> >(f, path, name, type);
      case bob::io::c128:
        return inner_get_scalar_attr<std::complex<double> >(f, path, name, type);
      case bob::io::c256:
        return inner_get_scalar_attr<std::complex<long double> >(f, path, name, type);
      default:
        break;
    }
  }

  //read as an numpy array
  bob::core::array::typeinfo atype;
  type.copy_to(atype);
  bob::python::py_array retval(atype);
  f.read_attribute(path, name, type, retval.ptr());
  return retval.pyobject();
}

static dict hdf5file_get_attributes(const bob::io::HDF5File& f, const std::string& path=".") {
  std::map<std::string, bob::io::HDF5Type> attributes;
  f.listAttributes(path, attributes);
  dict retval;
  for (std::map<std::string, bob::io::HDF5Type>::iterator k=attributes.begin(); k!=attributes.end(); ++k) {
    if (k->second.type() == bob::io::unsupported) {
      boost::format m("unsupported HDF5 data type detected for attribute '%s' - setting None");
      m % k->first;
      PYTHON_WARNING(UserWarning, m.str().c_str());
      retval[k->first] = object(); //None
    }
    else {
      retval[k->first] = inner_get_attr(f, path, k->first, k->second);
    }
  }
  return retval;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_get_attributes_overloads, hdf5file_get_attributes, 1, 2)

static object hdf5file_get_attribute(const bob::io::HDF5File& f, const std::string& name, const std::string& path=".") {
  bob::io::HDF5Type type;
  f.getAttributeType(path, name, type);
  if (type.type() == bob::io::unsupported) {
    boost::format m("unsupported HDF5 data type detected for attribute '%s' - returning None");
    m % name;
    PYTHON_WARNING(UserWarning, m.str().c_str());
    return object();
  }
  else {
    return inner_get_attr(f, path, name, type);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_get_attribute_overloads, hdf5file_get_attribute, 2, 3)

template <typename T>
static void inner_set_scalar_attr(bob::io::HDF5File& f, 
  const std::string& path, const std::string& name, const bob::io::HDF5Type& type,
  object obj) {
  T value = extract<T>(obj);
  f.write_attribute(path, name, type, static_cast<void*>(&value));
}

template <>
void inner_set_scalar_attr<std::string>(bob::io::HDF5File& f, 
  const std::string& path, const std::string& name, const bob::io::HDF5Type& type,
  object obj) {
  std::string value = extract<std::string>(obj);
  f.write_attribute(path, name, type, static_cast<const void*>(value.c_str()));
}

static void inner_set_attr(bob::io::HDF5File& f, const std::string& path,
    const std::string& name, const bob::io::HDF5Type& type, object obj,
    bool scalar) {

  //no error detection: this should be done before reaching this method

  if (scalar) { //write as a scalar
    switch(type.type()) {
      case bob::io::s:  
        return inner_set_scalar_attr<std::string>(f, path, name, type, obj);
      case bob::io::b:  
        return inner_set_scalar_attr<bool>(f, path, name, type, obj);
      case bob::io::i8:  
        return inner_set_scalar_attr<int8_t>(f, path, name, type, obj);
      case bob::io::i16: 
        return inner_set_scalar_attr<int16_t>(f, path, name, type, obj);
      case bob::io::i32:
        return inner_set_scalar_attr<int32_t>(f, path, name, type, obj);
      case bob::io::i64: 
        return inner_set_scalar_attr<int64_t>(f, path, name, type, obj);
      case bob::io::u8:  
        return inner_set_scalar_attr<uint8_t>(f, path, name, type, obj);
      case bob::io::u16: 
        return inner_set_scalar_attr<uint16_t>(f, path, name, type, obj);
      case bob::io::u32: 
        return inner_set_scalar_attr<uint32_t>(f, path, name, type, obj);
      case bob::io::u64: 
        return inner_set_scalar_attr<uint64_t>(f, path, name, type, obj);
      case bob::io::f32: 
        return inner_set_scalar_attr<float>(f, path, name, type, obj);
      case bob::io::f64: 
        return inner_set_scalar_attr<double>(f, path, name, type, obj);
      case bob::io::f128: 
        return inner_set_scalar_attr<long double>(f, path, name, type, obj);
      case bob::io::c64:
        return inner_set_scalar_attr<std::complex<float> >(f, path, name, type, obj);
      case bob::io::c128:
        return inner_set_scalar_attr<std::complex<double> >(f, path, name, type, obj);
      case bob::io::c256:
        return inner_set_scalar_attr<std::complex<long double> >(f, path, name, type, obj);
      default:
        break;
    }
  }

  else { //write as an numpy array
    bob::python::py_array retval(obj, object());
    f.write_attribute(path, name, type, retval.ptr());
  }
}

static void hdf5file_set_attributes(bob::io::HDF5File& f, dict attributes, const std::string& path=".") {
  object keys = attributes.iterkeys();
  for (int k=0; k<len(keys); ++k) {
    std::string key = extract<std::string>(keys[k]);
    bob::io::HDF5Type type;
    object obj = attributes[keys[k]];
    bool scalar = get_object_type(obj, type);
    inner_set_attr(f, path, key, type, obj, scalar);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_set_attributes_overloads, hdf5file_set_attributes, 2, 3)

static void hdf5file_set_attribute(bob::io::HDF5File& f, const std::string& key, object obj, const std::string& path=".") {
  bob::io::HDF5Type type;
  bool scalar = get_object_type(obj, type);
  inner_set_attr(f, path, key, type, obj, scalar);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_set_attribute_overloads, hdf5file_set_attribute, 3, 4)

static bool hdf5file_has_attribute(const bob::io::HDF5File& f, const std::string& name, const std::string& path=".") {
  return f.hasAttribute(path, name);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_has_attribute_overloads, hdf5file_has_attribute, 2, 3)

static void hdf5file_del_attribute(bob::io::HDF5File& f, const std::string& name, const std::string& path=".") {
  f.deleteAttribute(path, name);
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_del_attribute_overloads, hdf5file_del_attribute, 2, 3)

static void hdf5file_del_attributes(bob::io::HDF5File& f, const std::string& path=".") {
  std::map<std::string, bob::io::HDF5Type> attributes;
  f.listAttributes(path, attributes);
  for (std::map<std::string, bob::io::HDF5Type>::iterator k=attributes.begin(); k!=attributes.end(); ++k) {
    f.deleteAttribute(path, k->first);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(hdf5file_del_attributes_overloads, hdf5file_del_attributes, 1, 2)

void bind_io_hdf5() {
  class_<bob::io::HDF5File, boost::shared_ptr<bob::io::HDF5File> >("HDF5File", "A HDF5File allows users to read and write data from and to files containing standard bob binary coded data in HDF5 format. For an introduction to HDF5, please visit http://www.hdfgroup.org/HDF5.", no_init)
    .def(boost::python::init<const bob::io::HDF5File&>(boost::python::args("other"), "Generates a shallow copy of the already opened file."))
    .def("__init__", make_constructor(hdf5file_make_fromstr, default_call_policies(), (arg("filename"), arg("openmode_string") = "r")), "Opens a new file in one of these supported modes: 'r' (read-only), 'a' (read/write/append), 'w' (read/write/truncate) or 'x' (read/write/exclusive)")
    .def("cd", &bob::io::HDF5File::cd, (arg("self"), arg("path")), "Changes the current prefix path. When this object is started, the prefix path is empty, which means all following paths to data objects should be given using the full path. If you set this to a different value, it will be used as a prefix to any subsequent operation until you reset it. If path starts with '/', it is treated as an absolute path. '..' and '.' are supported. This object should be a std::string. If the value is relative, it is added to the current path. If it is absolute, it causes the prefix to be reset. Note all operations taking a relative path, following a cd(), will be considered relative to the value defined by the 'cwd' property of this object.")
    .def("has_group", &bob::io::HDF5File::hasGroup, (arg("self"), arg("path")), "Checks if a path exists inside a file - does not work for datasets, only for directories. If the given path is relative, it is take w.r.t. to the current working directory")
    .def("create_group", &bob::io::HDF5File::createGroup, (arg("self"), arg("path")), "Creates a new directory inside the file. A relative path is taken w.r.t. to the current directory. If the directory already exists (check it with hasGroup()), an exception will be raised.")
    .add_property("cwd", &bob::io::HDF5File::cwd)
    .def("__contains__", &bob::io::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")
    .def("has_key", &bob::io::HDF5File::contains, (arg("self"), arg("key")), "Returns True if the file contains an HDF5 dataset with a given path")
    .def("describe", &hdf5file_describe, (arg("self"), arg("key")), "If a given path to an HDF5 dataset exists inside the file, return a type description of objects recorded in such a dataset, otherwise, raises an exception. The returned value type is a tuple of tuples (HDF5Type, number-of-objects, expandible) describing the capabilities if the file is read using theses formats.")
    .def("unlink", &bob::io::HDF5File::unlink, (arg("self"), arg("key")), "If a given path to an HDF5 dataset exists inside the file, unlinks it. Please note this will note remove the data from the file, just make it inaccessible. If you wish to cleanup, save the reacheable objects from this file to another HDF5File object using copy(), for example.")
    .def("rename", &bob::io::HDF5File::rename, (arg("self"), arg("from"), arg("to")), "If a given path to an HDF5 dataset exists in the file, rename it")
    .def("keys", &hdf5file_paths, (arg("self"), arg("relative") = false), "Synonym for 'paths'")
    .def("paths", &hdf5file_paths, (arg("self"), arg("relative") = false), "Returns all paths to datasets available inside this file, stored under the current working directory. If relative is set to True, the returned paths are relative to the current working directory, otherwise they are asbolute.")
    .def("sub_groups", &hdf5file_sub_groups, (arg("self"), arg("relative") = false, arg("recursive") = true), "Returns all the subgroups (sub-directories) in the current file.")
    .def("copy", &bob::io::HDF5File::copy, (arg("self"), arg("file")), "Copies all accessible content to another HDF5 file")
    .def("read", &hdf5file_read, (arg("self"), arg("key")), "Reads the whole dataset in a single shot. Returns a single object with all contents.")
    .def("lread", (object(*)(bob::io::HDF5File&, const std::string&, int64_t))0, hdf5file_lread_overloads((arg("self"), arg("key"), arg("pos")=-1), "Reads a given position from the dataset. Returns a single object if 'pos' >= 0, otherwise a list by reading all objects in sequence."))
    .def("replace", &hdf5file_replace, (arg("self"), arg("path"), arg("pos"), arg("data")), "Modifies the value of a scalar/array inside a dataset in the file.\n\n" \
  "Keyword Parameters:\n\n" \
  "path\n" \
  "  This is the path to the HDF5 dataset to replace data at\n\n" \
  "pos\n" \
  "  This is the position we should replace\n\n" \
  "data\n" \
  "  This is the data that will be set on the position indicated")
    .def("append", &hdf5file_append, hdf5file_append_overloads((arg("self"), arg("path"), arg("data"), arg("compression")=0), "Appends a scalar or an array to a dataset. If the dataset does not yet exist, one is created with the type characteristics.\n\n" \
  "Keyword Parameters:\n\n" \
  "path\n" \
  "  This is the path to the HDF5 dataset to replace data at\n\n" \
  "data\n" \
  "  This is the data that will be set on the position indicated. It may be a simple python or numpy scalar (such as :py:class:`numpy.uint8`) or a :py:class:`numpy.ndarray` of any of the supported data types. You can also, optionally, set this to a list or tuple of scalars or arrays. This will cause this method to iterate over the elements and add each individually.\n\n" \
  "compresssion\n" \
  "  This parameter is effective when appending arrays. Set this to a number betwen 0 (default) and 9 (maximum) to compress the contents of this dataset. This setting is only effective if the dataset does not yet exist, otherwise, the previous setting is respected."))
    .def("set", &hdf5file_set, hdf5file_set_overloads((arg("self"), arg("path"), arg("data"), arg("compression")=0), "Sets the scalar or array at position 0 to the given value. This method is equivalent to checking if the scalar or array at position 0 exists and then replacing it. If the path does not exist, we append the new scalar or array.\n\n" \
  "Keyword Parameters:\n\n" \
  "path\n" \
  "  This is the path to the HDF5 dataset to replace data at\n\n" \
  "data\n" \
  "  This is the data that will be set on the position indicated. It may be a simple python or numpy scalar (such as :py:class:`numpy.uint8`) or a :py:class:`numpy.ndarray` of any of the supported data types. You can also, optionally, set this to an iterable of scalars or arrays. This will cause this method to collapse the whole iterable into a :py:class:`numpy.ndarray` and set that into the file.\n\n" \
  "compresssion\n" \
  "  This parameter is effective when setting arrays. Set this to a number betwen 0 (default) and 9 (maximum) to compress the contents of this dataset. This setting is only effective if the dataset does not yet exist, otherwise, the previous setting is respected."))
    // attribute manipulation
    .def("get_attributes", &hdf5file_get_attributes, hdf5file_get_attributes_overloads((arg("self"), arg("path")="."), "Returns a dictionary containing all attributes related to a particular (existing) path in this file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))
    .def("get_attribute", &hdf5file_get_attribute, hdf5file_get_attribute_overloads((arg("self"), arg("name"), arg("path")="."), "Returns an object representing an attribute attached to a particular (existing) path in this file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))
    .def("set_attributes", &hdf5file_set_attributes, hdf5file_set_attributes_overloads((arg("self"), arg("attrs"), arg("path")="."), "Sets attributes in a given (existing) path using a dictionary containing the names (keys) and values of those attributes. The path may point to a subdirectory or to a particular dataset. Only simple scalars (booleans, integers, floats and complex numbers) and arrays of those are supported at the time being. You can use :py:mod:`numpy` scalars to set values with arbitrary precision (e.g. :py:class:`numpy.uint8`). If the path does not exist, a RuntimeError is raised."))
    .def("set_attribute", &hdf5file_set_attribute, hdf5file_set_attribute_overloads((arg("self"), arg("name"), arg("value"), arg("path")="."), "Sets the attribute in a given (existing) path using the value provided. The path may point to a subdirectory or to a particular dataset. Only simple scalars (booleans, integers, floats and complex numbers) and arrays of those are supported at the time being. You can use :py:mod:`numpy` scalars to set values with arbitrary precision (e.g. :py:class:`numpy.uint8`). If the path does not exist, a RuntimeError is raised."))
    .def("has_attribute", &hdf5file_has_attribute, hdf5file_has_attribute_overloads((arg("self"), arg("name"), arg("path")="."), "Checks if given attribute exists in a given (existing) path. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))
    .def("delete_attribute", &hdf5file_del_attribute, hdf5file_del_attribute_overloads((arg("self"), arg("name"), arg("path")="."), "Deletes a given attribute associated to a (existing) path in the file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))
    .def("delete_attributes", &hdf5file_del_attributes, hdf5file_del_attributes_overloads((arg("self"), arg("path")="."), "Deletes **all** attributes associated to a (existing) path in the file. The path may point to a subdirectory or to a particular dataset. If the path does not exist, a RuntimeError is raised."))
    ;
}
