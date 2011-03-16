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

namespace Torch { namespace python {

  template <typename T, int N> 
    blitz::Array<T,N> add(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<T,N>(i1 + i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> add_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<T,N>(i1 + i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> sub(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<T,N>(i1 - i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> sub_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<T,N>(i1 - i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> mul(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<T,N>(i1 * i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> mul_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<T,N>(i1 * i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> div(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<T,N>(i1 / i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> div_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<T,N>(i1 / i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> mod(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<T,N>(i1 % i2); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> mod_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<T,N>(i1 % i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> lt(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 < i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> lt_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 < i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> le(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 <= i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> le_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 <= i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> gt(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 > i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> gt_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 > i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> ge(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 >= i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> ge_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 >= i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> ne(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 != i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> ne_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 != i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> eq(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 == i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> eq_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 == i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> and_(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 & i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> and_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 & i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> or_(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 | i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> or_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 | i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> xor_(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<bool,N>(i1 ^ i2); 
    }

  template <typename T, int N> 
    blitz::Array<bool,N> xor_c(blitz::Array<T,N>& i1, const T& i2) {
      return blitz::Array<bool,N>(i1 ^ i2); 
    }

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

  //Some special functions with 2 arguments
  template <typename T, int N> 
    blitz::Array<T,N> pow(blitz::Array<T,N>& i1, blitz::Array<T,N>& i2) {
      return blitz::Array<T,N>(blitz::pow(i1, i2)); 
    }

  template <typename T, int N> 
    blitz::Array<T,N> pow_c(blitz::Array<T,N>& i1, const T& i2) {
      blitz::Array<T,N> tmp(i1.shape());
      tmp = i2;
      return blitz::Array<T,N>(blitz::pow(i1, tmp)); 
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
      array.object()->def("__pow__", &pow_c<T,N>, "Computes self**argument.");
    }

  template <typename T, int N>
    void bind_common_arith (Torch::python::array<T,N>& array) {
      typedef typename Torch::python::array<T,N>::array_type array_type;
      typedef array_type& (array_type::*inplace_const_op)(const T&); 
      typedef array_type& (array_type::*inplace_array_op)(const array_type&);

      array.object()->def("__add__", &add<T,N>, "Adds two arrays element-wise"); 
      array.object()->def("__add__", &add_c<T,N>, "Adds an array with a constant element-wise"); 
      array.object()->def("__sub__", &sub<T,N>, "Subtracts two arrays element-wise"); 
      array.object()->def("__sub__", &sub_c<T,N>, "Subtracts an array with a constant element-wise"); 
      array.object()->def("__mul__", &mul<T,N>, "Multiplies two arrays element-wise");
      array.object()->def("__mul__", &mul_c<T,N>, "Multiplies an array with a constant element-wise"); 
      array.object()->def("__div__", &div<T,N>, "Divides two arrays element-wise"); 
      array.object()->def("__div__", &div_c<T,N>, "Divides an array with a constant element-wise"); 
      array.object()->def("__iadd__", (inplace_const_op)&array_type::operator+=, boost::python::return_self<>(), "Inplace addition with constant.");
      array.object()->def("__iadd__", (inplace_array_op)&array_type::operator+=, boost::python::return_self<>(), "Inplace addition with array, elementwise.");
      array.object()->def("__isub__", (inplace_const_op)&array_type::operator-=, boost::python::return_self<>(), "Inplace subtraction by constant.");
      array.object()->def("__isub__", (inplace_array_op)&array_type::operator-=, boost::python::return_self<>(), "Inplace subtraction by array, elementwise.");
      array.object()->def("__imul__", (inplace_const_op)&array_type::operator*=, boost::python::return_self<>(), "Inplace multiplication by constant");
      array.object()->def("__imul__", (inplace_array_op)&array_type::operator*=, boost::python::return_self<>(), "Inplace multiplication by array, elementwise.");
      array.object()->def("__idiv__", (inplace_const_op)&array_type::operator/=, boost::python::return_self<>(), "Inplace division by constant");
      array.object()->def("__idiv__", (inplace_array_op)&array_type::operator/=, boost::python::return_self<>(), "Inplace division by array, elementwise.");
      array.object()->def("__eq__", &eq<T,N>, "Compares two arrays element-wise"); 
      array.object()->def("__eq__", &eq_c<T,N>, "Compares an array to a constant element-wise"); 
      array.object()->def("__ne__", &ne<T,N>, "Compares two arrays element-wise"); 
      array.object()->def("__ne__", &ne_c<T,N>, "Compares an array to a constant element-wise"); 
      array.object()->def("__neg__", &neg<T,N>, "The negated values of the array element-wise"); 
    }

  template <typename T, int N>
    void bind_non_float_or_complex_arith (Torch::python::array<T,N>& array) {
      typedef typename Torch::python::array<T,N>::array_type array_type;
      typedef array_type& (array_type::*inplace_const_op)(const T&); 
      typedef array_type& (array_type::*inplace_array_op)(const array_type&);

      array.object()->def("__and__", &and_<T,N>, "Performs a bitwise and operation on two arrays, element-wise."); 
      array.object()->def("__and__", &and_c<T,N>, "Performs a bitwise and operation on two arrays, element-wise."); 
      array.object()->def("__or__", &or_<T,N>, "Performs a bitwise or operation on two arrays, element-wise."); 
      array.object()->def("__or__", &or_c<T,N>, "Performs a bitwise or operation on two arrays, element-wise."); 
      array.object()->def("__xor__", &xor_<T,N>, "Performs a bitwise xor operation on two arrays, element-wise."); 
      array.object()->def("__xor__", &xor_c<T,N>, "Performs a bitwise xor operation on two arrays, element-wise."); 
      array.object()->def("__ilshift__", (inplace_const_op)&array_type::operator<<=, boost::python::return_self<>(), "Inplace bitwise left-shift by constant.");
      array.object()->def("__ilshift__", (inplace_array_op)&array_type::operator<<=, boost::python::return_self<>(), "Inplace bitwise left-shift by array, elementwise.");
      array.object()->def("__irshift__", (inplace_const_op)&array_type::operator>>=, boost::python::return_self<>(), "Inplace bitwise right-shift by constant.");
      array.object()->def("__irshift__", (inplace_array_op)&array_type::operator>>=, boost::python::return_self<>(), "Inplace bitwise right-shift by array, elementwise.");
      array.object()->def("__iand__", (inplace_const_op)&array_type::operator&=, boost::python::return_self<>(), "Inplace bitwise and operation with constant.");
      array.object()->def("__iand__", (inplace_array_op)&array_type::operator&=, boost::python::return_self<>(), "Inplace bitwise and operation with array, elementwise.");
      array.object()->def("__ior__", (inplace_const_op)&array_type::operator|=, boost::python::return_self<>(), "Inplace bitwise or operation with constant.");
      array.object()->def("__ior__", (inplace_array_op)&array_type::operator|=, boost::python::return_self<>(), "Inplace bitwise or operation with array, elementwise.");
      array.object()->def("__ixor__", (inplace_const_op)&array_type::operator^=, boost::python::return_self<>(), "Inplace bitwise xor operation with constant.");
      array.object()->def("__ixor__", (inplace_array_op)&array_type::operator^=, boost::python::return_self<>(), "Inplace bitwise xor operation with array, elementwise.");
      array.object()->def("__invert__", &invert<T,N>, "The inverted values of the array element-wise"); 
      array.object()->def("__mod__", &mod<T,N>, "Executes the reminder of division between two arrays, element-wise.");
      array.object()->def("__mod__", &mod_c<T,N>, "Executes the reminder of division between two arrays, element-wise.");
      array.object()->def("__imod__", (inplace_const_op)&array_type::operator%=, boost::python::return_self<>(), "Inplace reminder of division by constant");
      array.object()->def("__imod__", (inplace_array_op)&array_type::operator%=, boost::python::return_self<>(), "Inplace reminder division by array, elementwise.");
    }

  template <typename T, int N>
    void bind_non_complex_arith (Torch::python::array<T,N>& array) {
      array.object()->def("__lt__", &lt<T,N>, "Compares two arrays element-wise"); 
      array.object()->def("__lt__", &lt_c<T,N>, "Compares an array to a constant element-wise"); 
      array.object()->def("__le__", &le<T,N>, "Compares two arrays element-wise"); 
      array.object()->def("__le__", &le_c<T,N>, "Compares an array to a constant element-wise"); 
      array.object()->def("__gt__", &gt<T,N>, "Compares two arrays element-wise"); 
      array.object()->def("__gt__", &gt_c<T,N>, "Compares an array to a constant element-wise"); 
      array.object()->def("__ge__", &ge<T,N>, "Compares two arrays element-wise"); 
      array.object()->def("__ge__", &ge_c<T,N>, "Compares an array to a constant element-wise"); 
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
