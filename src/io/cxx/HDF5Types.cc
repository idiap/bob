/**
 * @file io/cxx/HDF5Types.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A few helpers to handle HDF5 datasets in a more abstract way.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/format.hpp>
#include <sstream>
#include <boost/make_shared.hpp>

/**
 * MT "lock" support was only introduced in Boost 1.35. Before copying this
 * very ugly hack, make sure we are still using Boost 1.34. This will no longer
 * be the case starting January 2011.
 */
#include <boost/version.hpp>
#include <boost/thread/mutex.hpp>
#if ((BOOST_VERSION / 100) % 1000) > 34
#include <boost/thread/locks.hpp>
#else
#warning Disabling MT locks because Boost < 1.35!
#endif

#include "bob/core/logging.h"
#include "bob/io/HDF5Types.h"
#include "bob/io/HDF5Exception.h"

namespace io = bob::io;

const char* io::stringize (hdf5type t) {
  switch (t) {
    case io::s: 
      return "string";
    case io::b:
      return "bool";
    case io::i8:
      return "int8";
    case io::i16:
      return "int16";
    case io::i32:
      return "int32";
    case io::i64:
      return "int64";
    case io::u8:
      return "uint8";
    case io::u16:
      return "uint16";
    case io::u32:
      return "uint32";
    case io::u64:
      return "uint64";
    case io::f32:
      return "float32";
    case io::f64:
      return "float64";
    case io::f128:
      return "float128";
    case io::c64:
      return "complex64";
    case io::c128:
      return "complex128";
    case io::c256:
      return "complex256";
    case io::unsupported:
      return "unsupported";
  }
  return "unsupported"; ///< just to silence gcc
}

static herr_t walker(unsigned n, const H5E_error2_t *desc, void *cookie) {
  io::HDF5ErrorStack& stack = *(io::HDF5ErrorStack*)cookie;
  std::vector<std::string>& sv = stack.get();
  boost::format fmt("%s() @ %s+%d: %s");
  fmt % desc->func_name % desc->file_name % desc->line % desc->desc;
  sv.push_back(fmt.str());
  return 0;
}

static herr_t err_callback(hid_t stack, void* cookie) {
  io::HDF5ErrorStack& err_stack = *(io::HDF5ErrorStack*)cookie;
  if (!err_stack.muted()) H5Ewalk2(stack, H5E_WALK_DOWNWARD, walker, cookie);
  H5Eclear2(stack);
  return 0;
}

io::HDF5ErrorStack::HDF5ErrorStack ():
  m_stack(H5E_DEFAULT),
  m_muted(false),
  m_err(),
  m_func(0),
  m_client_data(0)
{
  H5Eget_auto2(m_stack, &m_func, &m_client_data);
  H5Eset_auto2(m_stack, err_callback, this);
}

io::HDF5ErrorStack::HDF5ErrorStack (hid_t stack):
  m_stack(stack),
  m_muted(false),
  m_err(),
  m_func(0),
  m_client_data(0)
{
  H5Eget_auto2(m_stack, &m_func, &m_client_data);
  H5Eset_auto2(m_stack, err_callback, this);
}

io::HDF5ErrorStack::~HDF5ErrorStack () {
  H5Eset_auto2(m_stack, m_func, m_client_data);
}

//creates a pointer to the default HDF5 error stack that is global to the
//application level.
const boost::shared_ptr<io::HDF5ErrorStack>
  io::DefaultHDF5ErrorStack(new HDF5ErrorStack());

io::HDF5Shape::HDF5Shape (size_t n):
  m_n(n),
  m_shape()
{
  if (n > MAX_HDF5SHAPE_SIZE) 
    throw std::length_error("maximum number of dimensions exceeded");
  for (size_t i=0; i<n; ++i) m_shape[i] = 0;
}

io::HDF5Shape::HDF5Shape ():
  m_n(0),
  m_shape()
{
}

io::HDF5Shape::HDF5Shape (const io::HDF5Shape& other):
  m_n(other.m_n),
  m_shape()
{
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
}

io::HDF5Shape::~HDF5Shape() {
}

io::HDF5Shape& io::HDF5Shape::operator= (const io::HDF5Shape& other) {
  m_n = other.m_n;
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

void io::HDF5Shape::copy(const io::HDF5Shape& other) {
  if (m_n <= other.m_n) { //I'm smaller or equal
    for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
  }
  else { //The other is smaller
    for (size_t i=0; i<other.m_n; ++i) m_shape[i] = other.m_shape[i];
  }
}

void io::HDF5Shape::reset() {
  m_n = 0;
}

io::HDF5Shape& io::HDF5Shape::operator <<= (size_t pos) {
  if (!m_n || !pos) return *this;
  for (size_t i=0; i<(m_n-pos); ++i) m_shape[i] = m_shape[i+pos];
  m_n -= pos;
  return *this;
}

io::HDF5Shape& io::HDF5Shape::operator >>= (size_t pos) {
  if (!pos) return *this;
  if ( (m_n + pos) > MAX_HDF5SHAPE_SIZE) 
    throw std::length_error("maximum number of dimensions will exceed");
  for (size_t i=(m_n+pos-1); i>(pos-1); --i) m_shape[i] = m_shape[i-1];
  for (size_t i=0; i<pos; ++i) m_shape[i] = 1;
  m_n += pos;
  return *this;
}

hsize_t io::HDF5Shape::product() const {
  hsize_t retval = 1;
  for (size_t i=0; i<m_n; ++i) retval *= m_shape[i];
  return retval;
}

bool io::HDF5Shape::operator== (const HDF5Shape& other) const {
  if (m_n != other.m_n) return false;
  for (size_t i=0; i<m_n; ++i) if (m_shape[i] != other[i]) return false;
  return true;
}

bool io::HDF5Shape::operator!= (const HDF5Shape& other) const {
  return !(*this == other);
}

std::string io::HDF5Shape::str () const {
  if (m_n == 0) return "";
  std::ostringstream retval("");
  retval << m_shape[0];
  for (size_t i=1; i<m_n; ++i) retval << ", " << m_shape[i];
  return retval.str();
}

/**
 * Deleter method for auto-destroyable HDF5 datatypes.
 */
static void delete_h5datatype (hid_t* p) {
  if (*p >= 0) {
    herr_t err = H5Tclose(*p); 
    if (err < 0) {
      bob::core::error << "H5Tclose() exited with an error (" << err << "). The stack trace follows:" << std::endl;
      bob::core::error << bob::io::format_hdf5_error() << std::endl;
    }
  }
  delete p; 
}

/**
 * Given a datatype which is a compound type, returns the std::complex<T>
 * hdf5type equivalent or raises.
 */
static io::hdf5type equivctype(const boost::shared_ptr<hid_t>& dt) {
  if (H5Tget_nmembers(*dt) != 2) throw io::HDF5UnsupportedTypeError(dt);

  //members have to:
  // 1. have names "real" and "imag"
  // 2. have class type H5T_FLOAT
  // 3. have equal size
  // 4. have a size of 4, 8 or 16 bytes

  // 1. 
  int real = H5Tget_member_index(*dt, "real");
  if (real < 0) {
    throw io::HDF5UnsupportedTypeError(dt);
  }
  int imag = H5Tget_member_index(*dt, "imag");
  if (imag < 0) {
    throw io::HDF5UnsupportedTypeError(dt);
  }

  // 2.
  if (H5Tget_member_class(*dt, real) != H5T_FLOAT) 
    throw io::HDF5UnsupportedTypeError(dt);
  if (H5Tget_member_class(*dt, imag) != H5T_FLOAT)
    throw io::HDF5UnsupportedTypeError(dt);

  // 3.
  boost::shared_ptr<hid_t> realid(new hid_t(-1), std::ptr_fun(delete_h5datatype));
  *realid = H5Tget_member_type(*dt, real);
  boost::shared_ptr<hid_t> imagid(new hid_t(-1), std::ptr_fun(delete_h5datatype));
  *imagid = H5Tget_member_type(*dt, imag);
  size_t realsize = H5Tget_size(*realid);
  size_t imagsize = H5Tget_size(*imagid);
  if (realsize != imagsize) {
    throw io::HDF5UnsupportedTypeError(dt);
  }

  // 4.
  switch (realsize) {
    case 4: //std::complex<float>
      return io::c64;
    case 8: //std::complex<double>
      return io::c128;
    case 16: //std::complex<double>
      return io::c256;
    default:
      break;
  }

  throw io::HDF5UnsupportedTypeError(dt);
}

/**
 * Checks if a given type can be read as boolean
 */
static void checkbool(const boost::shared_ptr<hid_t>& dt) {

  if (H5Tget_nmembers(*dt) != 2) throw io::HDF5UnsupportedTypeError(dt);
  
  int8_t value;
  herr_t status = H5Tget_member_value(*dt, 0, &value);
  if (status < 0) throw io::HDF5StatusError("H5Tget_member_value", status);
  bool next_is_false = false;
  if (value != 0) next_is_false = true;
  status = H5Tget_member_value(*dt, 1, &value);
  if (status < 0) throw io::HDF5StatusError("H5Tget_member_value", status);
  if (next_is_false) {
    if (value != 0) throw io::HDF5UnsupportedTypeError(dt);
  }
  else {
    if (value == 0) throw io::HDF5UnsupportedTypeError(dt);
  }
}

/**
 * Given a datatype, returns the supported type equivalent or raises
 */
static io::hdf5type get_datatype
(const boost::shared_ptr<hid_t>& dt) {
  H5T_class_t classtype = H5Tget_class(*dt);
  size_t typesize = H5Tget_size(*dt); ///< element size
  H5T_sign_t signtype = H5Tget_sign(*dt);
  
  //we only support little-endian byte-ordering
  H5T_order_t ordertype = H5Tget_order(*dt);

  //please note that checking compound types for hdf5 < 1.8.6 does not work.
# if H5_VERSION_GE(1,8,6)
  if (ordertype < 0) throw io::HDF5StatusError("H5Tget_order", ordertype);

  if (ordertype != H5T_ORDER_LE) {
    throw io::HDF5UnsupportedTypeError(dt);
  }
# else
  if ((ordertype >= 0) && (ordertype != H5T_ORDER_LE)) {
    throw io::HDF5UnsupportedTypeError(dt);
  }
# endif
  
  switch (classtype) {
    case H5T_ENUM:
      checkbool(dt);
      return io::b;
    case H5T_INTEGER:
      switch (typesize) {
        case 1: //int8 or uint8
          switch (signtype) {
            case H5T_SGN_NONE:
              return io::u8;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return io::i8;
            default:
              throw io::HDF5UnsupportedTypeError(dt);
          }
          break;
        case 2: //int16 or uint16
          switch (signtype) {
            case H5T_SGN_NONE:
              return io::u16;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return io::i16;
            default:
              throw io::HDF5UnsupportedTypeError(dt);
          }
          break;
        case 4: //int32 or uint32
          switch (signtype) {
            case H5T_SGN_NONE:
              return io::u32;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return io::i32;
            default:
              throw io::HDF5UnsupportedTypeError(dt);
          }
          break;
        case 8: //int64 or uint64
          switch (signtype) {
            case H5T_SGN_NONE:
              return io::u64;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return io::i64;
            default:
              throw io::HDF5UnsupportedTypeError(dt);
          }
          break;
        default:
          break;
      }
      break;
    case H5T_FLOAT:
      switch (typesize) {
        case 4: //float
          return io::f32;
        case 8: //double
          return io::f64;
        case 16: //long double
          return io::f128;
        default:
          break;
      }
      break;
    case H5T_STRING:
      return io::s;
    case H5T_COMPOUND: //complex
      return equivctype(dt);
    default:
      break;
  }
  throw io::HDF5UnsupportedTypeError(dt);
}

bool io::HDF5Type::compatible (const bob::core::array::typeinfo& value) const
{
  return *this == HDF5Type(value);
}

/**
 * Given a datatype, returns the supported HDF5 datatype equivalent or -1
 */
boost::shared_ptr<hid_t> io::HDF5Type::htype() const {
  switch (m_type) {
    case io::s:
      return boost::make_shared<hid_t>(H5T_STRING);
    case io::b:
      {
        //why? HDF5 is a C library and in C there is no boolean type
        //bottom-line => we have to define our own...

        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tenum_create(H5T_NATIVE_INT8);
        if (*retval < 0) throw io::HDF5StatusError("H5Tenum_create", *retval);
        int8_t val;
        herr_t status;
       
        //defines false
        val = 0;
        status = H5Tenum_insert(*retval, "false", &val);
        if (status < 0) throw io::HDF5StatusError("H5Tenum_insert", status);

        //defines true
        val = 1;
        status = H5Tenum_insert(*retval, "true",  &val);
        if (*retval < 0) throw io::HDF5StatusError("H5Tenum_insert", status);

        return retval;
      }
    case io::i8:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT8);
    case io::i16:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT16);
    case io::i32:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT32);
    case io::i64:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT64);
    case io::u8:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT8);
    case io::u16:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT16);
    case io::u32:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT32);
    case io::u64:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT64);
    case io::f32:
      return boost::make_shared<hid_t>(H5T_NATIVE_FLOAT);
    case io::f64:
      return boost::make_shared<hid_t>(H5T_NATIVE_DOUBLE);
    case io::f128:
      return boost::make_shared<hid_t>(H5T_NATIVE_LDOUBLE);
    case io::c64:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(float));
        if (*retval < 0) throw io::HDF5StatusError("H5Tcreate", *retval);
        herr_t status = H5Tinsert(*retval, "real", 0, H5T_NATIVE_FLOAT);
        if (status < 0) throw io::HDF5StatusError("H5Tinsert", status);
        status = H5Tinsert(*retval, "imag", sizeof(float), H5T_NATIVE_FLOAT);
        if (status < 0) throw io::HDF5StatusError("H5Tinsert", status);
        return retval;
      }
    case io::c128: 
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(double));
        if (*retval < 0) throw io::HDF5StatusError("H5Tcreate", *retval);
        herr_t status = H5Tinsert(*retval, "real", 0, H5T_NATIVE_DOUBLE);
        if (status < 0) throw io::HDF5StatusError("H5Tinsert", status);
        status = H5Tinsert(*retval, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
        if (status < 0) throw io::HDF5StatusError("H5Tinsert", status);
        return retval;
      }
    case io::c256:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(long double));
        if (*retval < 0) throw io::HDF5StatusError("H5Tcreate", *retval);
        herr_t status = H5Tinsert(*retval, "real", 0, H5T_NATIVE_LDOUBLE);
        if (status < 0) throw io::HDF5StatusError("H5Tinsert", status);
        status = H5Tinsert(*retval, "imag", sizeof(long double), H5T_NATIVE_LDOUBLE);
        if (status < 0) throw io::HDF5StatusError("H5Tinsert", status);
        return retval;
      }
    default:
      break;
  }
  throw io::HDF5UnsupportedTypeError();
}
  
#define DEFINE_SUPPORT(T,E) io::HDF5Type::HDF5Type(const T& value): \
    m_type(E), m_shape(1) { m_shape[0] = 1; }
DEFINE_SUPPORT(bool,io::b)
DEFINE_SUPPORT(int8_t,io::i8)
DEFINE_SUPPORT(int16_t,io::i16)
DEFINE_SUPPORT(int32_t,io::i32)
DEFINE_SUPPORT(int64_t,io::i64)
DEFINE_SUPPORT(uint8_t,io::u8)
DEFINE_SUPPORT(uint16_t,io::u16)
DEFINE_SUPPORT(uint32_t,io::u32)
DEFINE_SUPPORT(uint64_t,io::u64)
DEFINE_SUPPORT(float,io::f32)
DEFINE_SUPPORT(double,io::f64)
DEFINE_SUPPORT(long double,io::f128)
DEFINE_SUPPORT(std::complex<float>,io::c64)
DEFINE_SUPPORT(std::complex<double>,io::c128)
DEFINE_SUPPORT(std::complex<long double>,io::c256)
DEFINE_SUPPORT(std::string,io::s)
#undef DEFINE_SUPPORT

#define DEFINE_SUPPORT(T,E,N) io::HDF5Type::HDF5Type \
    (const blitz::Array<T,N>& value): \
      m_type(E), \
      m_shape(value.shape()) { \
        if (N > bob::core::array::N_MAX_DIMENSIONS_ARRAY) \
        throw io::HDF5UnsupportedDimensionError(N); \
      }

#define DEFINE_BZ_SUPPORT(T,E) \
  DEFINE_SUPPORT(T,E,1) \
  DEFINE_SUPPORT(T,E,2) \
  DEFINE_SUPPORT(T,E,3) \
  DEFINE_SUPPORT(T,E,4)

DEFINE_BZ_SUPPORT(bool,io::b)
DEFINE_BZ_SUPPORT(int8_t,io::i8)
DEFINE_BZ_SUPPORT(int16_t,io::i16)
DEFINE_BZ_SUPPORT(int32_t,io::i32)
DEFINE_BZ_SUPPORT(int64_t,io::i64)
DEFINE_BZ_SUPPORT(uint8_t,io::u8)
DEFINE_BZ_SUPPORT(uint16_t,io::u16)
DEFINE_BZ_SUPPORT(uint32_t,io::u32)
DEFINE_BZ_SUPPORT(uint64_t,io::u64)
DEFINE_BZ_SUPPORT(float,io::f32)
DEFINE_BZ_SUPPORT(double,io::f64)
DEFINE_BZ_SUPPORT(long double,io::f128)
DEFINE_BZ_SUPPORT(std::complex<float>,io::c64)
DEFINE_BZ_SUPPORT(std::complex<double>,io::c128)
DEFINE_BZ_SUPPORT(std::complex<long double>,io::c256)
#undef DEFINE_BZ_SUPPORT
#undef DEFINE_SUPPORT
      
io::HDF5Type::HDF5Type():
  m_type(io::unsupported),
  m_shape()
{
}

io::HDF5Type::HDF5Type(io::hdf5type type):
  m_type(type),
  m_shape(1)
{
  m_shape[0] = 1;
}

io::HDF5Type::HDF5Type(io::hdf5type type, const io::HDF5Shape& extents):
  m_type(type),
  m_shape(extents)
{
}

static io::hdf5type array_to_hdf5 (bob::core::array::ElementType eltype) {
  switch(eltype) {
    case bob::core::array::t_unknown:
      return io::unsupported;
    case bob::core::array::t_bool:
      return io::b;
    case bob::core::array::t_int8:
      return io::i8;
    case bob::core::array::t_int16:
      return io::i16;
    case bob::core::array::t_int32:
      return io::i32;
    case bob::core::array::t_int64:
      return io::i64;
    case bob::core::array::t_uint8:
      return io::u8;
    case bob::core::array::t_uint16:
      return io::u16;
    case bob::core::array::t_uint32:
      return io::u32;
    case bob::core::array::t_uint64:
      return io::u64;
    case bob::core::array::t_float32:
      return io::f32;
    case bob::core::array::t_float64:
      return io::f64;
    case bob::core::array::t_float128:
      return io::f128;
    case bob::core::array::t_complex64:
      return io::c64;
    case bob::core::array::t_complex128:
      return io::c128;
    case bob::core::array::t_complex256:
      return io::c256;
  }
  throw std::runtime_error("unsupported dtyle <=> hdf5 type conversion -- debug me");
}

io::HDF5Type::HDF5Type(const bob::core::array::typeinfo& ti): 
  m_type(array_to_hdf5(ti.dtype)),
  m_shape(ti.nd, ti.shape)
{
}

io::HDF5Type::HDF5Type(bob::core::array::ElementType eltype, 
    const HDF5Shape& extents): 
  m_type(array_to_hdf5(eltype)),
  m_shape(extents)
{
}

io::HDF5Type::HDF5Type(const boost::shared_ptr<hid_t>& type,
    const io::HDF5Shape& extents):
  m_type(get_datatype(type)),
  m_shape(extents)
{
}

io::HDF5Type::HDF5Type(const boost::shared_ptr<hid_t>& type):
  m_type(get_datatype(type)),
  m_shape(1) 
{
  m_shape[0] = 1; 
}

io::HDF5Type::HDF5Type(const HDF5Type& other):
  m_type(other.m_type),
  m_shape(other.m_shape)
{
}

io::HDF5Type::~HDF5Type() { }

io::HDF5Type& io::HDF5Type::operator= (const io::HDF5Type& other)
{
  m_type = other.m_type;
  m_shape = other.m_shape;
  return *this;
}

bool io::HDF5Type::operator== (const io::HDF5Type& other) const {
  return (m_type == other.m_type) && (m_shape == other.m_shape);
}

bool io::HDF5Type::operator!= (const io::HDF5Type& other) const {
  return !(*this == other);
}

std::string io::HDF5Type::str() const {
  boost::format retval("%s (%s)");
  retval % io::stringize(m_type) % m_shape.str();
  return retval.str();
}

bob::core::array::ElementType io::HDF5Type::element_type() const {
  switch (m_type) {
    case b:
      return bob::core::array::t_bool;
    case i8:
      return bob::core::array::t_int8;
    case i16:
      return bob::core::array::t_int16;
    case i32:
      return bob::core::array::t_int32;
    case i64:
      return bob::core::array::t_int64;
    case u8:
      return bob::core::array::t_uint8;
    case u16:
      return bob::core::array::t_uint16;
    case u32:
      return bob::core::array::t_uint32;
    case u64:
      return bob::core::array::t_uint64;
    case f32:
      return bob::core::array::t_float32;
    case f64:
      return bob::core::array::t_float64;
    case f128:
      return bob::core::array::t_float128;
    case c64:
      return bob::core::array::t_complex64;
    case c128:
      return bob::core::array::t_complex128;
    case c256:
      return bob::core::array::t_complex256;
    default:
      break;
  }
  return bob::core::array::t_unknown;
}

void io::HDF5Type::copy_to (bob::core::array::typeinfo& ti) const {
  ti.dtype = element_type();
  ti.nd = shape().n();
  if (ti.nd > (BOB_MAX_DIM+1)) {
    boost::format f("HDF5 type has more (%d) than the allowed maximum number of dimensions (%d)");
    f % ti.nd % (BOB_MAX_DIM+1);
    throw std::runtime_error(f.str());
  }
  for (size_t i=0; i<ti.nd; ++i) ti.shape[i] = shape()[i];
  ti.update_strides();
}
      
io::HDF5Descriptor::HDF5Descriptor(const HDF5Type& type, size_t size, 
          bool expand):
  type(type), 
  size(size),
  expandable(expand),
  hyperslab_start(type.shape().n()),
  hyperslab_count(type.shape())
{
}

io::HDF5Descriptor::HDF5Descriptor(const HDF5Descriptor& other):
  type(other.type),
  size(other.size),
  expandable(other.expandable),
  hyperslab_start(other.hyperslab_start),
  hyperslab_count(other.hyperslab_count)
{
}

io::HDF5Descriptor::~HDF5Descriptor() { }

io::HDF5Descriptor& io::HDF5Descriptor::operator=
(const io::HDF5Descriptor& other) {
  type = other.type;
  size = other.size;
  expandable = other.expandable;
  hyperslab_start = other.hyperslab_start;
  hyperslab_count = other.hyperslab_count;
  return *this;
}

io::HDF5Descriptor& io::HDF5Descriptor::subselect() {
  hyperslab_start >>= 1;
  hyperslab_count >>= 1;
  hyperslab_count[0] = 1;
  return *this;
}
