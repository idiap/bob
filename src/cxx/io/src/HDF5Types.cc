/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed 13 Apr 18:06:45 2011 
 *
 * @brief A few helpers to handle HDF5 datasets in a more abstract way.
 */

#include <boost/format.hpp>
#include <sstream>
#include <boost/make_shared.hpp>

#include "io/HDF5Types.h"
#include "io/HDF5Utils.h"
#include "io/HDF5Exception.h"

namespace io = Torch::io;

const char* io::stringize (hdf5type t) {
  switch (t) {
    case io::s: 
      return "string";
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

boost::shared_ptr<io::HDF5Error> io::HDF5Error::s_instance;

boost::shared_ptr<io::HDF5Error> io::HDF5Error::instance() {
  if (!s_instance) io::HDF5Error::s_instance.reset(new HDF5Error());
  return s_instance;
}

io::HDF5Error::HDF5Error (): m_error() {
}

io::HDF5Error::HDF5Error (hid_t stack): m_error(stack) {
}

io::HDF5Error::~HDF5Error() {
}

io::HDF5Shape::HDF5Shape (size_t n):
  m_n(n),
  m_shape(new hsize_t[n])
{
  for (size_t i=0; i<n; ++i) m_shape[i] = 0;
}

io::HDF5Shape::HDF5Shape ():
  m_n(0),
  m_shape()
{
}

io::HDF5Shape::HDF5Shape (const io::HDF5Shape& other):
  m_n(other.m_n),
  m_shape(new hsize_t[m_n])
{
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
}

io::HDF5Shape::~HDF5Shape() {
}

io::HDF5Shape& io::HDF5Shape::operator= (const io::HDF5Shape& other) {
  m_n = other.m_n;
  m_shape.reset(new hsize_t[m_n]);
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
  m_shape.reset();
}

io::HDF5Shape& io::HDF5Shape::operator <<= (size_t pos) {
  for (size_t i=0; i<(m_n-1); ++i) m_shape[i] = m_shape[i+1];
  m_n -= 1;
  return *this;
}

io::HDF5Shape& io::HDF5Shape::operator >>= (size_t pos) {
  for (size_t i=(m_n-1); i>0; --i) m_shape[i] = m_shape[i-1];
  m_shape[0] = 0;
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

/**
 * Given a datatype, returns the supported HDF5 datatype equivalent or -1
 */
boost::shared_ptr<hid_t> io::HDF5Type::htype() const {
  switch (m_type) {
    case io::s:
      return boost::make_shared<hid_t>(H5T_STRING);
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
        H5Tinsert(*retval, "real", 0, H5T_NATIVE_FLOAT);
        H5Tinsert(*retval, "imag", sizeof(float), H5T_NATIVE_FLOAT);
        return retval;
      }
    case io::c128: 
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(double));
        if (*retval < 0) throw io::HDF5StatusError("H5Tcreate", *retval);
        H5Tinsert(*retval, "real", 0, H5T_NATIVE_DOUBLE);
        H5Tinsert(*retval, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
        return retval;
      }
    case io::c256:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(long double));
        if (*retval < 0) throw io::HDF5StatusError("H5Tcreate", *retval);
        H5Tinsert(*retval, "real", 0, H5T_NATIVE_LDOUBLE);
        H5Tinsert(*retval, "imag", sizeof(long double), H5T_NATIVE_LDOUBLE);
        return retval;
      }
    default:
      break;
  }
  throw io::HDF5UnsupportedTypeError();
}
  
#define DEFINE_SUPPORT(T,E) io::HDF5Type::HDF5Type(const T& value): \
    m_type(E), \
    m_shape(1) { \
      m_shape[0] = 1; \
    }
DEFINE_SUPPORT(bool,io::u8)
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
        if (N > Torch::core::array::N_MAX_DIMENSIONS_ARRAY) \
        throw io::HDF5UnsupportedDimensionError(N); \
      }

#define DEFINE_BZ_SUPPORT(T,E) \
  DEFINE_SUPPORT(T,E,1) \
  DEFINE_SUPPORT(T,E,2) \
  DEFINE_SUPPORT(T,E,3) \
  DEFINE_SUPPORT(T,E,4)

DEFINE_BZ_SUPPORT(bool,io::u8)
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

io::HDF5Type::HDF5Type(const boost::shared_ptr<hid_t>& type,
    const io::HDF5Shape& extents):
  m_type(get_datatype(type)),
  m_shape(extents)
{
  m_shape <<= 1;
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

Torch::core::array::ElementType io::HDF5Type::element_type() const {
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
