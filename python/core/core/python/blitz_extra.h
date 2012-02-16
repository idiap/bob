/**
 * @file python/core/core/python/blitz_extra.h
 * @date Wed Mar 16 14:08:37 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Extra declarations to uniformize blitz::Array<> usage on different
 * architectures.
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

#ifndef BOB_CORE_PYTHON_BLITZ_EXTRA_H 
#define BOB_CORE_PYTHON_BLITZ_EXTRA_H

#include <blitz/array.h>
#include <stdint.h>

//some blitz extras for signed and unsigned 8 and 16-bit integers
namespace blitz {
  // abs(int8_t)
  template<>
  struct Fn_abs< int8_t > {
    typedef int8_t T_numtype1;
    typedef int8_t T_numtype;

    static T_numtype
      apply(T_numtype1 a) { return BZ_MATHFN_SCOPE(abs)(a); }

    template<typename T1>
      static void prettyPrint(BZ_STD_SCOPE(string) &str,
          prettyPrintFormat& format, const T1& t1) {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
      }
  };

  // abs(int16_t)
  template<> struct Fn_abs< int16_t > {
    typedef int16_t T_numtype1;
    typedef int16_t T_numtype;

    static T_numtype
      apply(T_numtype1 a) { return BZ_MATHFN_SCOPE(abs)(a); }

    template<typename T1>
      static void prettyPrint(BZ_STD_SCOPE(string) &str,
          prettyPrintFormat& format, const T1& t1) {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
      }
  };

#if !defined(__LP64__) || defined(__APPLE__)
  // abs(int64_t)
  template<> struct Fn_abs< int64_t > {
    typedef int64_t T_numtype1;
    typedef int64_t T_numtype;

    static T_numtype
      apply(T_numtype1 a) { return BZ_MATHFN_SCOPE(abs)(a); }

    template<typename T1>
      static void prettyPrint(BZ_STD_SCOPE(string) &str,
          prettyPrintFormat& format, const T1& t1) {
        str += "abs";
        str += "(";
        t1.prettyPrint(str, format);
        str += ")";
      }
  };
  // missing scalar ops
  BZ_DECLARE_ARRAY_ET_SCALAR_OPS(int64_t)
  BZ_DECLARE_ARRAY_ET_SCALAR_OPS(uint64_t)
#endif

}

#endif /* BOB_CORE_PYTHON_BLITZ_EXTRA_H */
