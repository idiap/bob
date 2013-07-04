/**
 * @file io/cxx/HDF5Types.cc
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief A few helpers to handle HDF5 datasets in a more abstract way.
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

#include <bob/core/logging.h>
#include <bob/io/HDF5Types.h>

const char* bob::io::stringize (hdf5type t) {
  switch (t) {
    case bob::io::s:
      return "string";
    case bob::io::b:
      return "bool";
    case bob::io::i8:
      return "int8";
    case bob::io::i16:
      return "int16";
    case bob::io::i32:
      return "int32";
    case bob::io::i64:
      return "int64";
    case bob::io::u8:
      return "uint8";
    case bob::io::u16:
      return "uint16";
    case bob::io::u32:
      return "uint32";
    case bob::io::u64:
      return "uint64";
    case bob::io::f32:
      return "float32";
    case bob::io::f64:
      return "float64";
    case bob::io::f128:
      return "float128";
    case bob::io::c64:
      return "complex64";
    case bob::io::c128:
      return "complex128";
    case bob::io::c256:
      return "complex256";
    case bob::io::unsupported:
      return "unsupported";
  }
  return "unsupported"; ///< just to silence gcc
}

static herr_t walker(unsigned n, const H5E_error2_t *desc, void *cookie) {
  bob::io::HDF5ErrorStack& stack = *(bob::io::HDF5ErrorStack*)cookie;
  std::vector<std::string>& sv = stack.get();
  boost::format fmt("%s() @ %s+%d: %s");
  fmt % desc->func_name % desc->file_name % desc->line % desc->desc;
  sv.push_back(fmt.str());
  return 0;
}

static herr_t err_callback(hid_t stack, void* cookie) {
  bob::io::HDF5ErrorStack& err_stack = *(bob::io::HDF5ErrorStack*)cookie;
  if (!err_stack.muted()) H5Ewalk2(stack, H5E_WALK_DOWNWARD, walker, cookie);
  H5Eclear2(stack);
  return 0;
}

bob::io::HDF5ErrorStack::HDF5ErrorStack ():
  m_stack(H5E_DEFAULT),
  m_muted(false),
  m_err(),
  m_func(0),
  m_client_data(0)
{
  H5Eget_auto2(m_stack, &m_func, &m_client_data);
  H5Eset_auto2(m_stack, err_callback, this);
}

bob::io::HDF5ErrorStack::HDF5ErrorStack (hid_t stack):
  m_stack(stack),
  m_muted(false),
  m_err(),
  m_func(0),
  m_client_data(0)
{
  H5Eget_auto2(m_stack, &m_func, &m_client_data);
  H5Eset_auto2(m_stack, err_callback, this);
}

bob::io::HDF5ErrorStack::~HDF5ErrorStack () {
  H5Eset_auto2(m_stack, m_func, m_client_data);
}

//creates a pointer to the default HDF5 error stack that is global to the
//application level.
const boost::shared_ptr<bob::io::HDF5ErrorStack>
  bob::io::DefaultHDF5ErrorStack(new HDF5ErrorStack());

bob::io::HDF5Shape::HDF5Shape (size_t n):
  m_n(n),
  m_shape()
{
  if (n > MAX_HDF5SHAPE_SIZE) {
    boost::format m("cannot create shape with %u dimensions, exceeding the maximum number of dimensions supported by this API (%u)");
    m % n % MAX_HDF5SHAPE_SIZE;
    throw std::runtime_error(m.str());
  }
  for (size_t i=0; i<n; ++i) m_shape[i] = 0;
}

bob::io::HDF5Shape::HDF5Shape ():
  m_n(0),
  m_shape()
{
}

bob::io::HDF5Shape::HDF5Shape (const bob::io::HDF5Shape& other):
  m_n(other.m_n),
  m_shape()
{
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
}

bob::io::HDF5Shape::~HDF5Shape() {
}

bob::io::HDF5Shape& bob::io::HDF5Shape::operator= (const bob::io::HDF5Shape& other) {
  m_n = other.m_n;
  for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

void bob::io::HDF5Shape::copy(const bob::io::HDF5Shape& other) {
  if (m_n <= other.m_n) { //I'm smaller or equal
    for (size_t i=0; i<m_n; ++i) m_shape[i] = other.m_shape[i];
  }
  else { //The other is smaller
    for (size_t i=0; i<other.m_n; ++i) m_shape[i] = other.m_shape[i];
  }
}

void bob::io::HDF5Shape::reset() {
  m_n = 0;
}

bob::io::HDF5Shape& bob::io::HDF5Shape::operator <<= (size_t pos) {
  if (!m_n || !pos) return *this;
  for (size_t i=0; i<(m_n-pos); ++i) m_shape[i] = m_shape[i+pos];
  m_n -= pos;
  return *this;
}

bob::io::HDF5Shape& bob::io::HDF5Shape::operator >>= (size_t pos) {
  if (!pos) return *this;
  if ( (m_n + pos) > MAX_HDF5SHAPE_SIZE) {
    boost::format m("if you shift right this shape by %u positions, you will exceed the maximum number of dimensions supported by this API (%u)");
    m % pos % MAX_HDF5SHAPE_SIZE;
    throw std::runtime_error(m.str());
  }
  for (size_t i=(m_n+pos-1); i>(pos-1); --i) m_shape[i] = m_shape[i-1];
  for (size_t i=0; i<pos; ++i) m_shape[i] = 1;
  m_n += pos;
  return *this;
}

hsize_t bob::io::HDF5Shape::product() const {
  hsize_t retval = 1;
  for (size_t i=0; i<m_n; ++i) retval *= m_shape[i];
  return retval;
}

bool bob::io::HDF5Shape::operator== (const HDF5Shape& other) const {
  if (m_n != other.m_n) return false;
  for (size_t i=0; i<m_n; ++i) if (m_shape[i] != other[i]) return false;
  return true;
}

bool bob::io::HDF5Shape::operator!= (const HDF5Shape& other) const {
  return !(*this == other);
}

std::string bob::io::HDF5Shape::str () const {
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
static bob::io::hdf5type equivctype(const boost::shared_ptr<hid_t>& dt) {
  if (H5Tget_nmembers(*dt) != 2) throw std::runtime_error("the internal HDF5 type is not supported by our HDF5 interface");

  //members have to:
  // 1. have names "real" and "imag"
  // 2. have class type H5T_FLOAT
  // 3. have equal size
  // 4. have a size of 4, 8 or 16 bytes

  // 1.
  int real = H5Tget_member_index(*dt, "real");
  if (real < 0) {
    throw std::runtime_error("the complex member index for `real' is not present on this HDF5 type");
  }
  int imag = H5Tget_member_index(*dt, "imag");
  if (imag < 0) {
    throw std::runtime_error("the complex member index for `imag' is not present on this HDF5 type");
  }

  // 2.
  if (H5Tget_member_class(*dt, real) != H5T_FLOAT)
    throw std::runtime_error("the raw type for member `real' on complex structure in HDF5 is not H5T_FLOAT as expected");
  if (H5Tget_member_class(*dt, imag) != H5T_FLOAT)
    throw std::runtime_error("the raw type for member `imag' on complex structure in HDF5 is not H5T_FLOAT as expected");

  // 3.
  boost::shared_ptr<hid_t> realid(new hid_t(-1), std::ptr_fun(delete_h5datatype));
  *realid = H5Tget_member_type(*dt, real);
  boost::shared_ptr<hid_t> imagid(new hid_t(-1), std::ptr_fun(delete_h5datatype));
  *imagid = H5Tget_member_type(*dt, imag);
  size_t realsize = H5Tget_size(*realid);
  size_t imagsize = H5Tget_size(*imagid);
  if (realsize != imagsize) {
    throw std::runtime_error("the sizes of the real and imaginary parts on HDF5 complex struct are not the same");
  }

  // 4.
  switch (realsize) {
    case 4: //std::complex<float>
      return bob::io::c64;
    case 8: //std::complex<double>
      return bob::io::c128;
    case 16: //std::complex<double>
      return bob::io::c256;
    default:
      break;
  }

  throw std::runtime_error("could not find the equivalent internal type for (supposedly) complex HDF5 structure");
}

/**
 * Checks if a given type can be read as boolean
 */
static void checkbool(const boost::shared_ptr<hid_t>& dt) {

  if (H5Tget_nmembers(*dt) != 2) {
    throw std::runtime_error("the number of enumeration members for the locally installed boolean type is not 2");
  }

  int8_t value;
  herr_t status = H5Tget_member_value(*dt, 0, &value);
  if (status < 0) {
    boost::format m("call to HDF5 C-function H5Tget_member_value() returned error %d. HDF5 error statck follows:\n%s");
    m % status % bob::io::format_hdf5_error();
    throw std::runtime_error(m.str());
  }
  bool next_is_false = false;
  if (value != 0) next_is_false = true;
  status = H5Tget_member_value(*dt, 1, &value);
  if (status < 0) {
    boost::format m("call to HDF5 C-function H5Tget_member_value() returned error %d. HDF5 error statck follows:\n%s");
    m % status % bob::io::format_hdf5_error();
    throw std::runtime_error(m.str());
  }
  if (next_is_false) {
    if (value != 0) {
      throw std::runtime_error("the attribution of false(0) or true(1) is messed up on the current data type, which is supposed to be a boolean");
    }
  }
  else {
    if (value == 0) {
      throw std::runtime_error("the attribution of false(0) or true(1) is messed up on the current data type, which is supposed to be a boolean");
    }
  }
}

/**
 * Given a datatype, returns the supported type equivalent or raises
 */
static bob::io::hdf5type get_datatype
(const boost::shared_ptr<hid_t>& dt) {
  H5T_class_t classtype = H5Tget_class(*dt);

  if (classtype == H5T_STRING) return bob::io::s; //no need to check further

  size_t typesize = H5Tget_size(*dt); ///< element size
  H5T_sign_t signtype = H5Tget_sign(*dt);

  //we only support little-endian byte-ordering
  H5T_order_t ordertype = H5Tget_order(*dt);

  //please note that checking compound types for hdf5 < 1.8.6 does not work.
# if H5_VERSION_GE(1,8,6)
  if (ordertype < 0) {
    boost::format m("call to HDF5 C-function H5Tget_order returned error %d. HDF5 error statck follows:\n%s");
    m % ordertype % bob::io::format_hdf5_error();
    throw std::runtime_error(m.str());
  }

  if (ordertype != H5T_ORDER_LE) {
    throw std::runtime_error("The endianness of datatype is not little-endian");
  }
# else
  if ((ordertype >= 0) && (ordertype != H5T_ORDER_LE)) {
    throw std::runtime_error("The endianness of datatype is not little-endian");
  }
# endif

  switch (classtype) {
    case H5T_ENUM:
      checkbool(dt);
      return bob::io::b;
    case H5T_INTEGER:
      switch (typesize) {
        case 1: //int8 or uint8
          switch (signtype) {
            case H5T_SGN_NONE:
              return bob::io::u8;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return bob::io::i8;
            default:
              throw std::runtime_error("HDF5 1-byte integer datatype (read from file) cannot be mapped into a C++ type supported by this API");
          }
          break;
        case 2: //int16 or uint16
          switch (signtype) {
            case H5T_SGN_NONE:
              return bob::io::u16;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return bob::io::i16;
            default:
              throw std::runtime_error("HDF5 2-byte integer datatype (read from file) cannot be mapped into a C++ type supported by this API");
          }
          break;
        case 4: //int32 or uint32
          switch (signtype) {
            case H5T_SGN_NONE:
              return bob::io::u32;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return bob::io::i32;
            default:
              throw std::runtime_error("HDF5 4-byte integer datatype (read from file) cannot be mapped into a C++ type supported by this API");
          }
          break;
        case 8: //int64 or uint64
          switch (signtype) {
            case H5T_SGN_NONE:
              return bob::io::u64;
            case H5T_SGN_2: //two's complement == "is signed" ;-)
              return bob::io::i64;
            default:
              throw std::runtime_error("HDF5 8-byte integer datatype (read from file) cannot be mapped into a C++ type supported by this API");
          }
          break;
        default:
          break;
      }
      break;
    case H5T_FLOAT:
      switch (typesize) {
        case 4: //float
          return bob::io::f32;
        case 8: //double
          return bob::io::f64;
        case 16: //long double
          return bob::io::f128;
        default:
          break;
      }
      break;
    case H5T_COMPOUND: //complex
      return equivctype(dt);
    default:
      break;
  }

  throw std::runtime_error("cannot handle HDF5 datatype on file using one of the native types supported by this API");
}

bool bob::io::HDF5Type::compatible (const bob::core::array::typeinfo& value) const
{
  return *this == HDF5Type(value);
}

/**
 * Given a datatype, returns the supported HDF5 datatype equivalent or -1
 */
boost::shared_ptr<hid_t> bob::io::HDF5Type::htype() const {
  switch (m_type) {
    case bob::io::s:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcopy(H5T_C_S1);
        if (*retval < 0) {
          boost::format m("call to HDF5 C-function H5Tcopy() returned error %d. HDF5 error statck follows:\n%s");
          m % *retval % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }

        //set string size
        herr_t status = H5Tset_size(*retval, m_shape[0]);
        if (status < 0) {
          boost::format m("Call to HDF5 C-function H5Tset_size() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }

        return retval;
      }
    case bob::io::b:
      {
        //why? HDF5 is a C library and in C there is no boolean type
        //bottom-line => we have to define our own...

        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tenum_create(H5T_NATIVE_INT8);
        if (*retval < 0) {
          boost::format m("call to HDF5 C-function H5Tenum_create() returned error %d. HDF5 error statck follows:\n%s");
          m % *retval % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        int8_t val;
        herr_t status;

        //defines false
        val = 0;
        status = H5Tenum_insert(*retval, "false", &val);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tenum_insert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }

        //defines true
        val = 1;
        status = H5Tenum_insert(*retval, "true",  &val);
        if (*retval < 0) {
          boost::format m("call to HDF5 C-function H5Tenum_insert() returned error %d. HDF5 error statck follows:\n%s");
          m % *retval % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }

        return retval;
      }
    case bob::io::i8:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT8);
    case bob::io::i16:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT16);
    case bob::io::i32:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT32);
    case bob::io::i64:
      return boost::make_shared<hid_t>(H5T_NATIVE_INT64);
    case bob::io::u8:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT8);
    case bob::io::u16:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT16);
    case bob::io::u32:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT32);
    case bob::io::u64:
      return boost::make_shared<hid_t>(H5T_NATIVE_UINT64);
    case bob::io::f32:
      return boost::make_shared<hid_t>(H5T_NATIVE_FLOAT);
    case bob::io::f64:
      return boost::make_shared<hid_t>(H5T_NATIVE_DOUBLE);
    case bob::io::f128:
      return boost::make_shared<hid_t>(H5T_NATIVE_LDOUBLE);
    case bob::io::c64:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(float));
        if (*retval < 0) {
          boost::format m("call to HDF5 C-function H5Tcreate() returned error %d. HDF5 error statck follows:\n%s");
          m % *retval % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        herr_t status = H5Tinsert(*retval, "real", 0, H5T_NATIVE_FLOAT);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tinsert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        status = H5Tinsert(*retval, "imag", sizeof(float), H5T_NATIVE_FLOAT);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tinsert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        return retval;
      }
    case bob::io::c128:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(double));
        if (*retval < 0) {
          boost::format m("call to HDF5 C-function H5Tcreate() returned error %d. HDF5 error statck follows:\n%s");
          m % *retval % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        herr_t status = H5Tinsert(*retval, "real", 0, H5T_NATIVE_DOUBLE);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tinsert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        status = H5Tinsert(*retval, "imag", sizeof(double), H5T_NATIVE_DOUBLE);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tinsert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        return retval;
      }
    case bob::io::c256:
      {
        boost::shared_ptr<hid_t> retval(new hid_t(-1),
            std::ptr_fun(delete_h5datatype));
        *retval = H5Tcreate(H5T_COMPOUND, 2*sizeof(long double));
        if (*retval < 0) {
          boost::format m("call to HDF5 C-function H5Tcreate() returned error %d. HDF5 error statck follows:\n%s");
          m % *retval % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        herr_t status = H5Tinsert(*retval, "real", 0, H5T_NATIVE_LDOUBLE);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tinsert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        status = H5Tinsert(*retval, "imag", sizeof(long double), H5T_NATIVE_LDOUBLE);
        if (status < 0) {
          boost::format m("call to HDF5 C-function H5Tinsert() returned error %d. HDF5 error statck follows:\n%s");
          m % status % bob::io::format_hdf5_error();
          throw std::runtime_error(m.str());
        }
        return retval;
      }
    default:
      break;
  }
  throw std::runtime_error("the C++ type you are trying to convert into a native HDF5 type is not supported by this API");
}

#define DEFINE_SUPPORT(T,E) bob::io::HDF5Type::HDF5Type(const T& value): \
    m_type(E), m_shape(1) { m_shape[0] = 1; }
DEFINE_SUPPORT(bool,bob::io::b)
DEFINE_SUPPORT(int8_t,bob::io::i8)
DEFINE_SUPPORT(int16_t,bob::io::i16)
DEFINE_SUPPORT(int32_t,bob::io::i32)
DEFINE_SUPPORT(int64_t,bob::io::i64)
DEFINE_SUPPORT(uint8_t,bob::io::u8)
DEFINE_SUPPORT(uint16_t,bob::io::u16)
DEFINE_SUPPORT(uint32_t,bob::io::u32)
DEFINE_SUPPORT(uint64_t,bob::io::u64)
DEFINE_SUPPORT(float,bob::io::f32)
DEFINE_SUPPORT(double,bob::io::f64)
DEFINE_SUPPORT(long double,bob::io::f128)
DEFINE_SUPPORT(std::complex<float>,bob::io::c64)
DEFINE_SUPPORT(std::complex<double>,bob::io::c128)
DEFINE_SUPPORT(std::complex<long double>,bob::io::c256)
#undef DEFINE_SUPPORT

bob::io::HDF5Type::HDF5Type(const std::string& value):
  m_type(bob::io::s),
  m_shape(1)
{
  m_shape[0] = value.size();
}

#define DEFINE_SUPPORT(T,E,N) bob::io::HDF5Type::HDF5Type \
    (const blitz::Array<T,N>& value): \
      m_type(E), \
      m_shape(value.shape()) { \
        if (N > bob::core::array::N_MAX_DIMENSIONS_ARRAY) {\
          boost::format m("you passed an array with %d dimensions, but this HDF5 API only supports arrays with up to %d dimensions"); \
          m % N % bob::core::array::N_MAX_DIMENSIONS_ARRAY; \
          throw std::runtime_error(m.str()); \
        } \
      }

#define DEFINE_BZ_SUPPORT(T,E) \
  DEFINE_SUPPORT(T,E,1) \
  DEFINE_SUPPORT(T,E,2) \
  DEFINE_SUPPORT(T,E,3) \
  DEFINE_SUPPORT(T,E,4)

DEFINE_BZ_SUPPORT(bool,bob::io::b)
DEFINE_BZ_SUPPORT(int8_t,bob::io::i8)
DEFINE_BZ_SUPPORT(int16_t,bob::io::i16)
DEFINE_BZ_SUPPORT(int32_t,bob::io::i32)
DEFINE_BZ_SUPPORT(int64_t,bob::io::i64)
DEFINE_BZ_SUPPORT(uint8_t,bob::io::u8)
DEFINE_BZ_SUPPORT(uint16_t,bob::io::u16)
DEFINE_BZ_SUPPORT(uint32_t,bob::io::u32)
DEFINE_BZ_SUPPORT(uint64_t,bob::io::u64)
DEFINE_BZ_SUPPORT(float,bob::io::f32)
DEFINE_BZ_SUPPORT(double,bob::io::f64)
DEFINE_BZ_SUPPORT(long double,bob::io::f128)
DEFINE_BZ_SUPPORT(std::complex<float>,bob::io::c64)
DEFINE_BZ_SUPPORT(std::complex<double>,bob::io::c128)
DEFINE_BZ_SUPPORT(std::complex<long double>,bob::io::c256)
#undef DEFINE_BZ_SUPPORT
#undef DEFINE_SUPPORT

bob::io::HDF5Type::HDF5Type():
  m_type(bob::io::unsupported),
  m_shape()
{
}

bob::io::HDF5Type::HDF5Type(bob::io::hdf5type type):
  m_type(type),
  m_shape(1)
{
  m_shape[0] = 1;
}

bob::io::HDF5Type::HDF5Type(bob::io::hdf5type type, const bob::io::HDF5Shape& extents):
  m_type(type),
  m_shape(extents)
{
}

static bob::io::hdf5type array_to_hdf5 (bob::core::array::ElementType eltype) {
  switch(eltype) {
    case bob::core::array::t_unknown:
      return bob::io::unsupported;
    case bob::core::array::t_bool:
      return bob::io::b;
    case bob::core::array::t_int8:
      return bob::io::i8;
    case bob::core::array::t_int16:
      return bob::io::i16;
    case bob::core::array::t_int32:
      return bob::io::i32;
    case bob::core::array::t_int64:
      return bob::io::i64;
    case bob::core::array::t_uint8:
      return bob::io::u8;
    case bob::core::array::t_uint16:
      return bob::io::u16;
    case bob::core::array::t_uint32:
      return bob::io::u32;
    case bob::core::array::t_uint64:
      return bob::io::u64;
    case bob::core::array::t_float32:
      return bob::io::f32;
    case bob::core::array::t_float64:
      return bob::io::f64;
    case bob::core::array::t_float128:
      return bob::io::f128;
    case bob::core::array::t_complex64:
      return bob::io::c64;
    case bob::core::array::t_complex128:
      return bob::io::c128;
    case bob::core::array::t_complex256:
      return bob::io::c256;
  }
  throw std::runtime_error("unsupported dtype <=> hdf5 type conversion -- FIXME");
}

bob::io::HDF5Type::HDF5Type(const bob::core::array::typeinfo& ti):
  m_type(array_to_hdf5(ti.dtype)),
  m_shape(ti.nd, ti.shape)
{
}

bob::io::HDF5Type::HDF5Type(bob::core::array::ElementType eltype,
    const HDF5Shape& extents):
  m_type(array_to_hdf5(eltype)),
  m_shape(extents)
{
}

bob::io::HDF5Type::HDF5Type(const boost::shared_ptr<hid_t>& type,
    const bob::io::HDF5Shape& extents):
  m_type(get_datatype(type)),
  m_shape(extents)
{
}

bob::io::HDF5Type::HDF5Type(const boost::shared_ptr<hid_t>& type):
  m_type(get_datatype(type)),
  m_shape(1)
{
  //strings have to be treated slightly differently
  if (H5Tget_class(*type) == H5T_STRING) m_shape[0] = H5Tget_size(*type);
  else m_shape[0] = 1;
}

bob::io::HDF5Type::HDF5Type(const HDF5Type& other):
  m_type(other.m_type),
  m_shape(other.m_shape)
{
}

bob::io::HDF5Type::~HDF5Type() { }

bob::io::HDF5Type& bob::io::HDF5Type::operator= (const bob::io::HDF5Type& other)
{
  m_type = other.m_type;
  m_shape = other.m_shape;
  return *this;
}

bool bob::io::HDF5Type::operator== (const bob::io::HDF5Type& other) const {
  return (m_type == other.m_type) && (m_shape == other.m_shape);
}

bool bob::io::HDF5Type::operator!= (const bob::io::HDF5Type& other) const {
  return !(*this == other);
}

std::string bob::io::HDF5Type::str() const {
  boost::format retval("%s (%s)");
  retval % bob::io::stringize(m_type) % m_shape.str();
  return retval.str();
}

bob::core::array::ElementType bob::io::HDF5Type::element_type() const {
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
    case s:
      throw std::runtime_error("Cannot convert HDF5 string type to an element type to be used in blitz::Array's - FIXME: something is wrong in the logic");
    default:
      break;
  }
  return bob::core::array::t_unknown;
}

void bob::io::HDF5Type::copy_to (bob::core::array::typeinfo& ti) const {
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

bob::io::HDF5Descriptor::HDF5Descriptor(const HDF5Type& type, size_t size,
          bool expand):
  type(type),
  size(size),
  expandable(expand),
  hyperslab_start(type.shape().n()),
  hyperslab_count(type.shape())
{
}

bob::io::HDF5Descriptor::HDF5Descriptor(const HDF5Descriptor& other):
  type(other.type),
  size(other.size),
  expandable(other.expandable),
  hyperslab_start(other.hyperslab_start),
  hyperslab_count(other.hyperslab_count)
{
}

bob::io::HDF5Descriptor::~HDF5Descriptor() { }

bob::io::HDF5Descriptor& bob::io::HDF5Descriptor::operator=
(const bob::io::HDF5Descriptor& other) {
  type = other.type;
  size = other.size;
  expandable = other.expandable;
  hyperslab_start = other.hyperslab_start;
  hyperslab_count = other.hyperslab_count;
  return *this;
}

bob::io::HDF5Descriptor& bob::io::HDF5Descriptor::subselect() {
  hyperslab_start >>= 1;
  hyperslab_count >>= 1;
  hyperslab_count[0] = 1;
  return *this;
}

std::string bob::io::format_hdf5_error() {
  const std::vector<std::string>& stack = bob::io::DefaultHDF5ErrorStack->get();
  std::ostringstream retval;
  std::string prefix(" ");
  if (stack.size()) retval << prefix << stack[0];
  for (size_t i=1; i<stack.size(); ++i)
    retval << std::endl << prefix << stack[i];
  bob::io::DefaultHDF5ErrorStack->clear();
  return retval.str();
}
