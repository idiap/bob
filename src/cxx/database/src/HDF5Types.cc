/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 13 Apr 18:06:45 2011 
 *
 * @brief A few helpers to handle HDF5 datasets in a more abstract way.
 */

#include <boost/format.hpp>
#include <sstream>
#include <boost/make_shared.hpp>

#include "database/HDF5Types.h"
#include "database/HDF5Exception.h"

namespace db = Torch::database;

const char* db::stringize (hdf5type t) {
  switch (t) {
    case db::s: 
      return "string";
    case db::i8:
      return "int8";
    case db::i16:
      return "int16";
    case db::i32:
      return "int32";
    case db::i64:
      return "int64";
    case db::u8:
      return "uint8";
    case db::u16:
      return "uint16";
    case db::u32:
      return "uint32";
    case db::u64:
      return "uint64";
    case db::f32:
      return "float32";
    case db::f64:
      return "float64";
    case db::f128:
      return "float128";
    case db::c64:
      return "complex64";
    case db::c128:
      return "complex128";
    case db::c256:
      return "complex256";
    case db::unsupported:
      return "unsupported";
  }
  return "unsupported"; ///< just to silence gcc
}

static herr_t walker(unsigned n, const H5E_error2_t *desc, void *cookie) {
  db::HDF5ErrorStack& stack = *(db::HDF5ErrorStack*)cookie;
  std::vector<std::string>& sv = stack.get();
  boost::format fmt("%s() @ %s+%d: %s");
  fmt % desc->func_name % desc->file_name % desc->line % desc->desc;
  sv.push_back(fmt.str());
  return 0;
}

static herr_t err_callback(hid_t stack, void* cookie) {
  db::HDF5ErrorStack& err_stack = *(db::HDF5ErrorStack*)cookie;
  if (!err_stack.muted()) H5Ewalk(stack, H5E_WALK_DOWNWARD, walker, cookie);
  H5Eclear(stack);
  return 0;
}

db::HDF5ErrorStack::HDF5ErrorStack ():
  m_stack(H5E_DEFAULT),
  m_muted(false),
  m_err(),
  m_func(0),
  m_client_data(0)
{
  H5Eget_auto(m_stack, &m_func, &m_client_data);
  H5Eset_auto(m_stack, err_callback, this);
}

db::HDF5ErrorStack::HDF5ErrorStack (hid_t stack):
  m_stack(stack),
  m_muted(false),
  m_err(),
  m_func(0),
  m_client_data(0)
{
  H5Eget_auto(m_stack, &m_func, &m_client_data);
  H5Eset_auto(m_stack, err_callback, this);
}

db::HDF5ErrorStack::~HDF5ErrorStack () {
  H5Eset_auto(m_stack, m_func, m_client_data);
}

boost::shared_ptr<db::HDF5Error> db::HDF5Error::s_instance;

boost::shared_ptr<db::HDF5Error> db::HDF5Error::instance() {
  if (!s_instance) db::HDF5Error::s_instance.reset(new HDF5Error());
  return s_instance;
}

db::HDF5Error::HDF5Error (): m_error() {
}

db::HDF5Error::HDF5Error (hid_t stack): m_error(stack) {
}

db::HDF5Error::~HDF5Error() {
}

db::HDF5Shape::HDF5Shape (size_t n):
  m_n(n),
  m_shape(new hsize_t[n])
{
  for (size_t i=0; i<n; ++i) m_shape[i] = 0;
}

db::HDF5Shape::HDF5Shape ():
  m_n(0),
  m_shape()
{
}

db::HDF5Shape::HDF5Shape (const db::HDF5Shape& other):
  m_n(other.m_n),
  m_shape(new hsize_t[m_n])
{
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
}

db::HDF5Shape::~HDF5Shape() {
}

db::HDF5Shape& db::HDF5Shape::operator= (const db::HDF5Shape& other) {
  m_n = other.m_n;
  m_shape.reset(new hsize_t[m_n]);
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

void db::HDF5Shape::copy(const db::HDF5Shape& other) {
  if (m_n <= other.m_n) { //I'm smaller or equal
    for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
  }
  else { //The other is smaller
    for (size_t i=0; i<other.m_n; ++i) m_shape[i] = other.m_shape[i];
  }
}

void db::HDF5Shape::reset() {
  m_n = 0;
  m_shape.reset();
}

db::HDF5Shape& db::HDF5Shape::operator <<= (size_t pos) {
  for (size_t i=0; i<(m_n-1); ++i) m_shape[i] = m_shape[i+1];
  m_n -= 1;
  return *this;
}

db::HDF5Shape& db::HDF5Shape::operator >>= (size_t pos) {
  for (size_t i=(m_n-1); i>0; --i) m_shape[i] = m_shape[i-1];
  m_shape[0] = 0;
  return *this;
}

hsize_t db::HDF5Shape::product() const {
  hsize_t retval = 1;
  for (size_t i=0; i<m_n; ++i) retval *= m_shape[i];
  return retval;
}

bool db::HDF5Shape::operator== (const HDF5Shape& other) const {
  if (m_n != other.m_n) return false;
  for (size_t i=0; i<m_n; ++i) if (m_shape[i] != other[i]) return false;
  return true;
}

bool db::HDF5Shape::operator!= (const HDF5Shape& other) const {
  return !(*this == other);
}

std::string db::HDF5Shape::str () const {
  std::ostringstream retval("");
  if (!m_n) return retval.str();
  retval << m_shape[0];
  for (size_t i=1; i<m_n; ++i) retval << ", " << m_shape[i];
  return retval.str();
}

/**
 * Deleter method for auto-destroyable HDF5 datatypes.
 */
static void delete_h5datatype (hid_t* p) {
  if (*p >= 0) H5Tclose(*p); 
  delete p; 
  p=0; 
}

/**
 * Given a datatype which is a compound type, returns the std::complex<T>
 * hdf5type equivalent or raises.
 */
static db::hdf5type equivctype(const boost::shared_ptr<hid_t>& dt) {
  if (H5Tget_nmembers(*dt) != 2) throw db::HDF5UnsupportedTypeError(dt);

  //members have to:
  // 1. have names "real" and "imag"
  // 2. have class type H5T_FLOAT
  // 3. have equal size
  // 4. have a size of 4, 8 or 16 bytes

  // 1. 
  int real = H5Tget_member_index(*dt, "real");
  if (real < 0) {
    throw db::HDF5UnsupportedTypeError(dt);
  }
  int imag = H5Tget_member_index(*dt, "imag");
  if (imag < 0) {
    throw db::HDF5UnsupportedTypeError(dt);
  }

  // 2.
  if (H5Tget_member_class(*dt, real) != H5T_FLOAT) 
    throw db::HDF5UnsupportedTypeError(dt);
  if (H5Tget_member_class(*dt, imag) != H5T_FLOAT)
    throw db::HDF5UnsupportedTypeError(dt);

  // 3.
  boost::shared_ptr<hid_t> realid(new hid_t(-1), std::ptr_fun(delete_h5datatype));
  *realid = H5Tget_member_type(*dt, real);
  boost::shared_ptr<hid_t> imagid(new hid_t(-1), std::ptr_fun(delete_h5datatype));
  *imagid = H5Tget_member_type(*dt, imag);
  size_t realsize = H5Tget_size(*realid);
  size_t imagsize = H5Tget_size(*imagid);
  if (realsize != imagsize) {
    throw db::HDF5UnsupportedTypeError(dt);
  }

  // 4.
  switch (realsize) {
    case 4: //std::complex<float>
      return db::c64;
    case 8: //std::complex<double>
      return db::c128;
    case 16: //std::complex<double>
      return db::c256;
    default:
      break;
  }

  throw db::HDF5UnsupportedTypeError(dt);
}

/**
 * Given a datatype, returns the supported type equivalent or raises
 */
static db::hdf5type get_datatype
(const boost::shared_ptr<hid_t>& dt) {
  H5T_class_t classtype = H5Tget_class(*dt);
  size_t typesize = H5Tget_size(*dt); ///< element size
  H5T_sign_t signtype = H5Tget_sign(*dt);
  
  //we only support little-endian byte-ordering
  H5T_order_t ordertype = H5Tget_order(*dt);
  if (ordertype != H5T_ORDER_LE) {
    throw db::HDF5UnsupportedTypeError(dt);
  }
  
  switch (classtype) {
    case H5T_INTEGER:
      switch (typesize) {
        case 1: //int8 or uint8
          switch (signtype) {
            case H5T_SGN_NONE:
              return db::u8;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return db::i8;
            default:
              throw db::HDF5UnsupportedTypeError(dt);
          }
          break;
        case 2: //int16 or uint16
          switch (signtype) {
            case H5T_SGN_NONE:
              return db::u16;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return db::i16;
            default:
              throw db::HDF5UnsupportedTypeError(dt);
          }
          break;
        case 4: //int32 or uint32
          switch (signtype) {
            case H5T_SGN_NONE:
              return db::u32;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return db::i32;
            default:
              throw db::HDF5UnsupportedTypeError(dt);
          }
          break;
        case 8: //int64 or uint64
          switch (signtype) {
            case H5T_SGN_NONE:
              return db::u64;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return db::i64;
            default:
              throw db::HDF5UnsupportedTypeError(dt);
          }
          break;
        default:
          break;
      }
      break;
    case H5T_FLOAT:
      switch (typesize) {
        case 4: //float
          return db::f32;
        case 8: //double
          return db::f64;
        case 16: //long double
          return db::f128;
        default:
          break;
      }
      break;
    case H5T_STRING:
      return db::s;
    case H5T_COMPOUND: //complex
      return equivctype(dt);
    default:
      break;
  }
  throw db::HDF5UnsupportedTypeError(dt);
}

/**
 * Given a datatype, returns the supported HDF5 datatype equivalent or -1
 */
boost::shared_ptr<hid_t> db::HDF5Type::htype() const {
  switch (m_type) {
    case db::s:
      return boost::make_shared<hid_t>(H5T_STRING);
    case db::i8:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT8);
    case db::i16:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT16);
    case db::i32:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT32);
    case db::i64:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT64);
    case db::u8:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT8);
    case db::u16:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT16);
    case db::u32:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT32);
    case db::u64:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT64);
    case db::f32:
      return boost::make_shared<hid_t>(H5T_NATIVE_FLOAT);
    case db::f64:
      return boost::make_shared<hid_t>(H5T_NATIVE_DOUBLE);
    case db::f128:
      return boost::make_shared<hid_t>(H5T_NATIVE_LDOUBLE);
    case db::c64:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(float));
        if (*retval < 0) throw db::HDF5StatusError("H5Tcreate", *retval);
        H5Tinsert(*retval, "real", 0, H5T_NATIVE_FLOAT);
        H5Tinsert(*retval, "imag", sizeof(float), H5T_NATIVE_FLOAT);
        return retval;
      }
    case db::c128: 
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(double));
        if (*retval < 0) throw db::HDF5StatusError("H5Tcreate", *retval);
        H5Tinsert(*retval, "real", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(*retval, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
        return retval;
      }
    case db::c256:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(long double));
        if (*retval < 0) throw db::HDF5StatusError("H5Tcreate", *retval);
        H5Tinsert(*retval, "real", 0, H5T_NATIVE_LDOUBLE);
        H5Tinsert(*retval, "imag", sizeof(long double), H5T_NATIVE_LDOUBLE);
        return retval;
      }
    default:
      break;
  }
  throw db::HDF5UnsupportedTypeError();
}
  
#define DEFINE_SUPPORT(T,E) db::HDF5Type::HDF5Type(const T& value): \
    m_type(E), \
    m_shape(1) { \
      m_shape[0] = 1; \
    }
DEFINE_SUPPORT(bool,db::u8)
DEFINE_SUPPORT(int8_t,db::i8)
DEFINE_SUPPORT(int16_t,db::i16)
DEFINE_SUPPORT(int32_t,db::i32)
DEFINE_SUPPORT(int64_t,db::i64)
DEFINE_SUPPORT(uint8_t,db::u8)
DEFINE_SUPPORT(uint16_t,db::u16)
DEFINE_SUPPORT(uint32_t,db::u32)
DEFINE_SUPPORT(uint64_t,db::u64)
DEFINE_SUPPORT(float,db::f32)
DEFINE_SUPPORT(double,db::f64)
DEFINE_SUPPORT(long double,db::f128)
DEFINE_SUPPORT(std::complex<float>,db::c64)
DEFINE_SUPPORT(std::complex<double>,db::c128)
DEFINE_SUPPORT(std::complex<long double>,db::c256)
DEFINE_SUPPORT(std::string,db::s)
#undef DEFINE_SUPPORT

#define DEFINE_SUPPORT(T,E,N) db::HDF5Type::HDF5Type \
    (const blitz::Array<T,N>& value): \
      m_type(E), \
      m_shape(value.shape()) { \
        if (N > Torch::core::array::N_MAX_DIMENSIONS_ARRAY) \
        throw db::HDF5UnsupportedDimensionError(N); \
      }

#define DEFINE_BZ_SUPPORT(T,E) \
  DEFINE_SUPPORT(T,E,1) \
  DEFINE_SUPPORT(T,E,2) \
  DEFINE_SUPPORT(T,E,3) \
  DEFINE_SUPPORT(T,E,4)

DEFINE_BZ_SUPPORT(bool,db::u8)
DEFINE_BZ_SUPPORT(int8_t,db::i8)
DEFINE_BZ_SUPPORT(int16_t,db::i16)
DEFINE_BZ_SUPPORT(int32_t,db::i32)
DEFINE_BZ_SUPPORT(int64_t,db::i64)
DEFINE_BZ_SUPPORT(uint8_t,db::u8)
DEFINE_BZ_SUPPORT(uint16_t,db::u16)
DEFINE_BZ_SUPPORT(uint32_t,db::u32)
DEFINE_BZ_SUPPORT(uint64_t,db::u64)
DEFINE_BZ_SUPPORT(float,db::f32)
DEFINE_BZ_SUPPORT(double,db::f64)
DEFINE_BZ_SUPPORT(long double,db::f128)
DEFINE_BZ_SUPPORT(std::complex<float>,db::c64)
DEFINE_BZ_SUPPORT(std::complex<double>,db::c128)
DEFINE_BZ_SUPPORT(std::complex<long double>,db::c256)
#undef DEFINE_BZ_SUPPORT
#undef DEFINE_SUPPORT

db::HDF5Type::HDF5Type(const boost::shared_ptr<hid_t>& type,
    const db::HDF5Shape& extents):
  m_type(get_datatype(type)),
  m_shape(extents)
{
  m_shape <<= 1;
}

db::HDF5Type::HDF5Type(const HDF5Type& other):
  m_type(other.m_type),
  m_shape(other.m_shape)
{
}

db::HDF5Type::~HDF5Type() { }

db::HDF5Type& db::HDF5Type::operator= (const db::HDF5Type& other)
{
  m_type = other.m_type;
  m_shape = other.m_shape;
  return *this;
}

bool db::HDF5Type::operator== (const db::HDF5Type& other) const {
  return (m_type == other.m_type) && (m_shape == other.m_shape);
}

bool db::HDF5Type::operator!= (const db::HDF5Type& other) const {
  return !(*this == other);
}

std::string db::HDF5Type::str() const {
  boost::format retval("%s (%s)");
  retval % db::stringize(m_type) % m_shape.str();
  return retval.str();
}

Torch::core::array::ElementType db::HDF5Type::element_type() const {
  switch (m_type) {
    case i8:
      return Torch::core::array::t_int8;
    case i16:
      return Torch::core::array::t_int16;
    case i32:
      return Torch::core::array::t_int32;
    case i64:
      return Torch::core::array::t_int64;
    case u8:
      return Torch::core::array::t_uint8;
    case u16:
      return Torch::core::array::t_uint16;
    case u32:
      return Torch::core::array::t_uint32;
    case u64:
      return Torch::core::array::t_uint64;
    case f32:
      return Torch::core::array::t_float32;
    case f64:
      return Torch::core::array::t_float64;
    case f128:
      return Torch::core::array::t_float128;
    case c64:
      return Torch::core::array::t_complex64;
    case c128:
      return Torch::core::array::t_complex128;
    case c256:
      return Torch::core::array::t_complex256;
    default:
      break;
  }
  return Torch::core::array::t_unknown;
}
