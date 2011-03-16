/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Wed  9 Mar 19:52:50 2011 
 *
 * @brief Extra declarations to uniformize blitz::Array<> usage on different
 * architectures.
 */

#ifndef TORCH_CORE_PYTHON_BLITZ_EXTRA_H 
#define TORCH_CORE_PYTHON_BLITZ_EXTRA_H

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

#endif /* TORCH_CORE_PYTHON_BLITZ_EXTRA_H */
