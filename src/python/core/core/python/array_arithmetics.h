/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 12 Mar 22:07:38 2011 
 *
 * @brief Declares all sorts of arith arithmetic operations
 */

#ifndef TORCH_PYTHON_CORE_ARRAY_ARITHMETICS_H 
#define TORCH_PYTHON_CORE_ARRAY_ARITHMETICS_H

#include <boost/python.hpp>
#include <blitz/array.h>

#include "core/python/array_base.h"
#include "core/python/blitz_extra.h"

#define ARITH_COMP_OP(T,N,NAME,OP) template <typename T, int N> boost::python::object NAME(const blitz::Array<T,N>& i1, boost::python::object i2) { \
      boost::python::extract<T> try_t(i2); \
      if (try_t.check()) return boost::python::object(blitz::Array<bool,N>(i1 OP try_t())); \
      boost::python::extract<const blitz::Array<T,N>&> try_bz(i2); \
      if (try_bz.check()) return boost::python::object(blitz::Array<bool,N>(i1 OP try_bz())); \
      PyErr_SetString(PyExc_TypeError, "arithmetic operation against this blitz::Array<> requires a constant or another blitz::Array<>"); \
      boost::python::throw_error_already_set(); \
      return boost::python::object(); \
    }

#define ARITH_OP(T,N,NAME,OP) template <typename T, int N> boost::python::object NAME(const blitz::Array<T,N>& i1, boost::python::object i2) { \
      boost::python::extract<T> try_t(i2); \
      if (try_t.check()) return boost::python::object(blitz::Array<T,N>(i1 OP try_t())); \
      boost::python::extract<const blitz::Array<T,N>&> try_bz(i2); \
      if (try_bz.check()) return boost::python::object(blitz::Array<T,N>(i1 OP try_bz())); \
      PyErr_SetString(PyExc_TypeError, "arithmetic operation against this blitz::Array<> requires a constant or another blitz::Array<>"); \
      boost::python::throw_error_already_set(); \
      return boost::python::object(); \
    }

#define ARITH_IOP(T,N,NAME,OP) template <typename T, int N> boost::python::object NAME(blitz::Array<T,N>& i1, boost::python::object i2) { \
      boost::python::extract<T> try_t(i2); \
      if (try_t.check()) return boost::python::object(blitz::Array<T,N>(i1 OP try_t())); \
      boost::python::extract<const blitz::Array<T,N>&> try_bz(i2); \
      if (try_bz.check()) return boost::python::object(blitz::Array<T,N>(i1 OP try_bz())); \
      PyErr_SetString(PyExc_TypeError, "arithmetic operation against this blitz::Array<> requires a constant or another blitz::Array<>"); \
      boost::python::throw_error_already_set(); \
      return boost::python::object(); \
    }

namespace Torch { namespace python {

  ARITH_OP(T,N,bzadd,+)
  ARITH_OP(T,N,bzsub,-)
  ARITH_OP(T,N,bzmul,*)
  ARITH_OP(T,N,bzdiv,/)
  ARITH_OP(T,N,bzmod,%)

  ARITH_COMP_OP(T,N,bzlt,<)
  ARITH_COMP_OP(T,N,bzle,<=)
  ARITH_COMP_OP(T,N,bzgt,>)
  ARITH_COMP_OP(T,N,bzge,>=)
  ARITH_COMP_OP(T,N,bzne,!=)
  ARITH_COMP_OP(T,N,bzeq,==)

  ARITH_OP(T,N,bzand,&)
  ARITH_OP(T,N,bzor,|)
  ARITH_OP(T,N,bzxor,^)

  ARITH_IOP(T,N,bziadd,+=)
  ARITH_IOP(T,N,bzisub,-=)
  ARITH_IOP(T,N,bzimul,*=)
  ARITH_IOP(T,N,bzidiv,/=)
  ARITH_IOP(T,N,bzimod,%=)
  ARITH_IOP(T,N,bzilshift,<<=)
  ARITH_IOP(T,N,bzirshift,>>=)
  ARITH_IOP(T,N,bziand,&=)
  ARITH_IOP(T,N,bzior,|=)
  ARITH_IOP(T,N,bzixor,^=)

  template <typename T, int N> 
    blitz::Array<T,N> invert(blitz::Array<T,N>& i) {
      return blitz::Array<T,N>(~i); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> neg(blitz::Array<T,N>& i) {
      return blitz::Array<T,N>(-i); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> abs(blitz::Array<T,N>& i) {
      return blitz::Array<T,N>(blitz::abs(i)); 
    }

  //The power function requires a special treatment
  template <typename T, int N> 
    boost::python::object pow(blitz::Array<T,N>& i1, boost::python::object i2) {
      boost::python::extract<T> try_t(i2);
      if (try_t.check()) {
        blitz::Array<T,N> tmp(i1.shape());
        tmp = try_t();
        return boost::python::object(blitz::Array<T,N>(blitz::pow(i1, tmp)));
      }
      boost::python::extract<const blitz::Array<T,N>&> try_bz(i2);
      if (try_bz.check()) 
        return boost::python::object(blitz::Array<T,N>(blitz::pow(i1, try_bz())));

      PyErr_SetString(PyExc_TypeError, "arithmetic operation against this blitz::Array<> requires a constant or another blitz::Array<>");
      boost::python::throw_error_already_set();
      return boost::python::object();
    }

  template <typename T, int N>
    void bind_non_bool_or_uint_arith (Torch::python::array<T,N>& array) {
      array.object()->def("__abs__", &abs<T,N>, "Absolute value");
    }

  /**
   * Methods that operate on everything that is float or complex 
   */
  template <typename T, int N>
    void bind_float_complex_arith (Torch::python::array<T,N>& array) {
      array.object()->def("__pow__", &pow<T,N>, "Computes self**argument.");
    }

  template <typename T, int N>
    void bind_common_arith (Torch::python::array<T,N>& array) {
      array.object()->def("__add__", &bzadd<T,N>);
      array.object()->def("__sub__", &bzsub<T,N>);
      array.object()->def("__mul__", &bzmul<T,N>);
      array.object()->def("__div__", &bzdiv<T,N>);
      array.object()->def("__eq__", &bzeq<T,N>);
      array.object()->def("__ne__", &bzne<T,N>);
      array.object()->def("__iadd__", &bziadd<T,N>);
      array.object()->def("__isub__", &bzisub<T,N>);
      array.object()->def("__imul__", &bzimul<T,N>);
      array.object()->def("__idiv__", &bzidiv<T,N>);
      array.object()->def("__neg__", &neg<T,N>);
    }

  template <typename T, int N>
    void bind_non_float_or_complex_arith (Torch::python::array<T,N>& array) {
      array.object()->def("__mod__", &bzmod<T,N>);
      array.object()->def("__and__", &bzand<T,N>);
      array.object()->def("__or__", &bzor<T,N>);
      array.object()->def("__xor__", &bzxor<T,N>);
      array.object()->def("__imod__", &bzimod<T,N>);
      array.object()->def("__ilshift__", &bzilshift<T,N>);
      array.object()->def("__irshift__", &bzirshift<T,N>);
      array.object()->def("__iand__", &bziand<T,N>);
      array.object()->def("__ior__", &bzior<T,N>);
      array.object()->def("__ixor__", &bzixor<T,N>);
      array.object()->def("__invert__", &invert<T,N>);
    }

  template <typename T, int N>
    void bind_non_complex_arith (Torch::python::array<T,N>& array) {
      array.object()->def("__lt__", &bzlt<T,N>);
      array.object()->def("__le__", &bzle<T,N>);
      array.object()->def("__gt__", &bzgt<T,N>);
      array.object()->def("__ge__", &bzge<T,N>);
    }

  template <typename T, int N> void bind_bool_arith (Torch::python::array<T,N>& array) {
    bind_common_arith(array);
    bind_non_float_or_complex_arith(array);
    bind_non_complex_arith(array);
  }

  template <typename T, int N> void bind_int_arith (Torch::python::array<T,N>& array) {
    bind_common_arith(array);
    bind_non_bool_or_uint_arith(array);
    bind_non_float_or_complex_arith(array);
    bind_non_complex_arith(array);
  }

  template <typename T, int N> void bind_uint_arith (Torch::python::array<T,N>& array) {
    bind_common_arith(array);
    bind_non_float_or_complex_arith(array);
    bind_non_complex_arith(array);
  }

  template <typename T, int N> void bind_float_arith (Torch::python::array<T,N>& array) {
    bind_common_arith(array);
    bind_non_bool_or_uint_arith(array);
    bind_non_complex_arith(array);
    bind_float_complex_arith(array);
  }

  template <typename T, int N> void bind_complex_arith (Torch::python::array<T,N>& array) {
    bind_common_arith(array);
    bind_non_bool_or_uint_arith(array);
    bind_float_complex_arith(array);
  }

}}

#endif /* TORCH_PYTHON_CORE_ARRAY_ARITHMETICS_H */
