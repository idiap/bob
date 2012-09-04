/**
 * @file cxx/core/src/array_type.cc
 * @date Sat Apr 9 18:10:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Some type-related array utilities
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

#include "bob/core/array_type.h"
#include "bob/core/Exception.h"
#include <boost/format.hpp>

namespace ca = bob::core::array;

static const char* t_bool_string = "bool";
static const char* t_int8_string = "int8";
static const char* t_int16_string = "int16";
static const char* t_int32_string = "int32";
static const char* t_int64_string = "int64";
static const char* t_uint8_string = "uint8";
static const char* t_uint16_string = "uint16";
static const char* t_uint32_string = "uint32";
static const char* t_uint64_string = "uint64";
static const char* t_float32_string = "float32";
static const char* t_float64_string = "float64";
static const char* t_float128_string = "float128";
static const char* t_complex64_string = "complex64";
static const char* t_complex128_string = "complex128";
static const char* t_complex256_string = "complex256";
static const char* t_unknown_string = "unknown";

size_t ca::getElementSize(ElementType t) {
  switch(t) {
    case ca::t_bool:
      return sizeof(bool);
    case ca::t_int8:
      return sizeof(int8_t);
    case ca::t_int16:
      return sizeof(int16_t);
    case ca::t_int32:
      return sizeof(int32_t);
    case ca::t_int64:
      return sizeof(int64_t);
    case ca::t_uint8:
      return sizeof(uint8_t);
    case ca::t_uint16:
      return sizeof(uint16_t);
    case ca::t_uint32:
      return sizeof(uint32_t);
    case ca::t_uint64:
      return sizeof(uint64_t);
    case ca::t_float32:
      return sizeof(float);
    case ca::t_float64:
      return sizeof(double);
    case ca::t_float128:
      return sizeof(long double);
    case ca::t_complex64:
      return sizeof(std::complex<float>);
    case ca::t_complex128:
      return sizeof(std::complex<double>);
    case ca::t_complex256:
      return sizeof(std::complex<long double>);
    default:
      {
        boost::format m("unsupported element type (%d)");
        m % (int)t;
        throw std::runtime_error(m.str());
      }
  }
}

const char* ca::stringize(ElementType t) {
  switch(t) {
    case ca::t_bool: 
      return t_bool_string;
    case ca::t_int8:
      return t_int8_string;
    case ca::t_int16:
      return t_int16_string;
    case ca::t_int32:
      return t_int32_string;
    case ca::t_int64:
      return t_int64_string;
    case ca::t_uint8:
      return t_uint8_string;
    case ca::t_uint16:
      return t_uint16_string;
    case ca::t_uint32:
      return t_uint32_string;
    case ca::t_uint64:
      return t_uint64_string;
    case ca::t_float32:
      return t_float32_string;
    case ca::t_float64:
      return t_float64_string;
    case ca::t_float128:
      return t_float128_string;
    case ca::t_complex64:
      return t_complex64_string;
    case ca::t_complex128:
      return t_complex128_string;
    case ca::t_complex256:
      return t_complex256_string;
    default:
      return t_unknown_string;
  }
}

ca::ElementType ca::unstringize(const char* s) {
  std::string sc(s);
  if (sc == t_bool_string) return ca::t_bool;
  if (sc == t_int8_string) return ca::t_int8;
  if (sc == t_int16_string) return ca::t_int16;
  if (sc == t_int32_string) return ca::t_int32;
  if (sc == t_int64_string) return ca::t_int64;
  if (sc == t_uint8_string) return ca::t_uint8;
  if (sc == t_uint16_string) return ca::t_uint16;
  if (sc == t_uint32_string) return ca::t_uint32;
  if (sc == t_uint64_string) return ca::t_uint64;
  if (sc == t_float32_string) return ca::t_float32;
  if (sc == t_float64_string) return ca::t_float64;
  if (sc == t_float128_string) return ca::t_float128;
  if (sc == t_complex64_string) return ca::t_complex64;
  if (sc == t_complex128_string) return ca::t_complex128;
  if (sc == t_complex256_string) return ca::t_complex256;
  return ca::t_unknown;
}
