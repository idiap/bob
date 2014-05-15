/**
 * @date Fri Jul 29 22:22:48 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Constructs to allow the concatenation of blitz::Arrays along
 * configurable dimensions in a generic way and optimally avoiding excessive
 * copying.
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_ARRAY_CAT_H
#define BOB_CORE_ARRAY_CAT_H

#include <blitz/array.h>
#include <vector>

namespace bob { namespace core { namespace array {
  /**
   * @ingroup CORE_ARRAY
   * @{
   */

  /**
   * @brief Copies the data of one array into the other, specifying a precise
   * position and a dimension along which the copy will take place. No checks
   * are done, just trust the user.
   *
   * Requires: Arrays have the same shape, except for the dimension in which
   * the copy will occur.
   *
   * Requires: The destination array should have enough space allocated.
   *
   * @return An array of the same type, but with the two input arrays
   * concatenated along the given dimension.
   */
  template <typename T, int N>
    void dcopy_(const blitz::Array<T,N>& source, blitz::Array<T,N>& dest,
        int D, int pos) {
      blitz::RectDomain<N> domain = dest.domain();
      domain.lbound(D) = pos;
      domain.ubound(D) = pos + source.extent(D) - 1;
      dest(domain) = source; //copy
    }

  /**
   * @brief Copies the data of one array into the other, specifying a precise
   * position and a dimension along which the copy will take place.
   *
   * Requires: Arrays have the same shape, except for the concatenation
   * dimension.
   *
   * Requires: The destination array should have enough space allocated.
   *
   * @return An array of the same type, but with the two input arrays
   * concatenated along the given dimension.
   */
  template <typename T, int N>
    void dcopy(const blitz::Array<T,N>& source, blitz::Array<T,N>& dest,
        int D, int pos) {

      if ( D >= N )
        throw std::range_error("Copy dimension greater or equal total number of dimensions");

      //checks arrays are compatible
      blitz::TinyVector<int,N> v1 = source.shape();
      blitz::TinyVector<int,N> v2 = dest.shape();
      v1(D) = v2(D) = 0;
      if ( blitz::any(v1 != v2) )
        throw std::runtime_error("Arrays are not compatible for copy along the given dimension");

      //checks destination has enough room
      if ( (source.extent(D) + pos) > dest.extent(D) )
        throw std::range_error("Destination array does not hold enough space");

      dcopy_(source, dest, D, pos);
    }

  /**
   * @brief Copies the data of array a into the destination array d without
   * checking the shape. "a" has N-1 dimensions and is copied along dimension
   * "D" in "d" at position "pos".
   */
  template <typename T, int N> struct copy__ {

      static void f(const blitz::Array<T,N-1>& a,
          blitz::Array<T,N>& d, int D, int pos) {
        //generic implementation: will not compile!
        blitz::TinyVector<int,N> lowerBounds = 0;
        blitz::TinyVector<int,N> upperBounds = d.shape();
        upperBounds -= 1;
        lowerBounds(D) = upperBounds(D) = pos;
        blitz::RectDomain<N> domain(lowerBounds, upperBounds);
        d(domain) = a;
      }

    };

  /**
   * @brief Here is the 2D specializations that do compile.
   */
  template <typename T> struct copy__<T,2> {

      static void f(const blitz::Array<T,1>& a,
          blitz::Array<T,2>& d, int D, int pos) {
        blitz::Range all = blitz::Range::all();
        switch (D) {
          case 0:
            d(pos, all) = a;
            break;
          case 1:
            d(all, pos) = a;
            break;
        }
      }

  };

  /**
   * @brief Here is the 3D specializations that do compile.
   */
  template <typename T> struct copy__<T,3> {

      static void f(const blitz::Array<T,2>& a,
          blitz::Array<T,3>& d, int D, int pos) {
        blitz::Range all = blitz::Range::all();
        switch (D) {
          case 0:
            d(pos, all, all) = a;
            break;
          case 1:
            d(all, pos, all) = a;
            break;
          case 2:
            d(all, all, pos) = a;
            break;
        }
      }

  };

  /**
   * @brief Here is the 4D specializations that do compile.
   */
  template <typename T> struct copy__<T,4> {

      static void f(const blitz::Array<T,3>& a,
          blitz::Array<T,4>& d, int D, int pos) {
        blitz::Range all = blitz::Range::all();
        switch (D) {
          case 0:
            d(pos, all, all, all) = a;
            break;
          case 1:
            d(all, pos, all, all) = a;
            break;
          case 2:
            d(all, all, pos, all) = a;
            break;
          case 3:
            d(all, all, all, pos) = a;
            break;
        }
      }

  };

  /**
   * @brief Copies the data of "source" along the given dimension of "dest".
   * Special case: source has dimension N-1. Does not check any of the
   * requirements (trust the user).
   *
   * Requires: Arrays have the same shape, except for the copy dimension.
   *
   * Requires: The destination array should have enough space allocated.
   *
   * @return An array of the same type, but with the two input arrays
   * concatenated along the given dimension.
   */
  template <typename T, int N> void dcopy_(const blitz::Array<T,N-1>& source,
      blitz::Array<T,N>& dest, int D, int pos) {
      copy__<T,N>::f(source, dest, D, pos);
    }

  /**
   * @brief Copies the data of "source" along the given dimension of "dest".
   * Special case: source has dimension N-1.
   *
   * Requires: Arrays have the same shape, except for the copy dimension.
   *
   * Requires: The destination array should have enough space allocated.
   *
   * @return An array of the same type, but with the two input arrays
   * concatenated along the given dimension.
   */
  template <typename T, int N>
    void dcopy(const blitz::Array<T,N-1>& source, blitz::Array<T,N>& dest,
        int D, int pos) {

      if ( D >= N )
        throw std::range_error("Copy dimension greater or equal total number of dimensions");

      //checks arrays are compatible
      for (int k=0, l=0; k<N; ++k) { //k => dest; l => source
        if (k != D) { //skip comparison if k == D
          if (dest.shape()(k) != source.shape()(l++)) { //increment l
            throw std::runtime_error("Arrays are not compatible for copy along the given dimension");
          }
        }
      }

      //checks destination has enough room
      if ( pos >= dest.extent(D) )
        throw std::range_error("Destination array does not hold enough space");

      copy__<T,N>::f(source, dest, D, pos);
    }

  /**
   * @brief Concatenates a bunch of arrays with the same shape together, along
   * dimension D. Does not check the user input.
   *
   * Requires: The source and destination shapes are identical except along
   * dimension D.
   *
   * Requires: The destination array has enough space.
   */
  template <typename T, int N>
    void cat_(const std::vector<blitz::Array<T,N> >& source,
        blitz::Array<T,N>& dest, int D) {
      int pos = 0;
      for (size_t i=0; i<source.size(); ++i) {
        dcopy_(source[i], dest, D, pos);
        pos += source[i].extent(D);
      }
    }

  /**
   * @brief Concatenates a bunch of arrays with the same shape together, along
   * dimension D.
   *
   * Requires: The source and destination shapes are identical except along
   * dimension D.
   *
   * Requires: The destination array has enough space.
   */
  template <typename T, int N>
    void cat(const std::vector<blitz::Array<T,N> >& source,
        blitz::Array<T,N>& dest, int D) {
      int pos = 0;
      for (size_t i=0; i<source.size(); ++i) {
        dcopy(source[i], dest, D, pos);
        pos += source[i].extent(D);
      }
    }

  /**
   * @brief Stacks a bunch of arrays with N-1 dimensions together, along the
   * first dimension of the destination array. Does not check the user input.
   *
   * Note: If you want to stack along a different dimension, just transpose the
   * result or give a transposed destination.
   *
   * Requires: The source and destination shapes are identical except along
   * dimension D.
   *
   * Requires: The destination array has enough space.
   */
  template <typename T, int N>
    void stack_(const std::vector<blitz::Array<T,N-1> >& source,
        blitz::Array<T,N>& dest) {
      int pos = 0;
      for (size_t i=0; i<source.size(); ++i) {
        dcopy_(source[i], dest, 0, pos++);
      }
    }

  /**
   * @brief Stacks a bunch of arrays with N-1 dimensions together, along the
   * first dimension of the destination array.
   *
   * Note: If you want to stack along a different dimension, just transpose the
   * result or give a transposed destination.
   *
   * Requires: The source and destination shapes are identical except along
   * dimension D.
   *
   * Requires: The destination array has enough space.
   */
  template <typename T, int N>
    void stack(const std::vector<blitz::Array<T,N-1> >& source,
        blitz::Array<T,N>& dest) {
      int pos = 0;
      for (size_t i=0; i<source.size(); ++i) {
        dcopy(source[i], dest, 0, pos++);
      }
    }

  /**
   * @}
   */
}}}

#endif /* BOB_CORE_ARRAY_CAT_H */
