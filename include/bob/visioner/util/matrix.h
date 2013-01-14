/**
 * @file bob/visioner/util/matrix.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#ifndef BOB_VISIONER_MATRIX_H
#define BOB_VISIONER_MATRIX_H

#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>
#include <blitz/array.h>

#include "bob/visioner/util/iterators.h"

namespace bob { namespace visioner {

  /**
   * 2D matrix using an 1D STL vector.
   */
  template <typename T> class Matrix {

    public: //api

      // Constructor
      Matrix(size_t rows = 0, size_t cols = 0, T fillValue = T())
        :	m_rows(rows), m_cols(cols), m_data(rows * cols, fillValue) { }

      // Starts from an existing pointer
      Matrix(size_t rows, size_t cols, const T* data)
        : m_rows(rows), m_cols(cols) {
          size_t size = m_rows * m_cols;
          m_data.resize(size);
          std::copy(data, data+size, m_data.begin());
        }

      // Starts from an existing pointer
      template <typename U> Matrix(const blitz::Array<U,2>& other)
        : m_rows(other.extent(0)), m_cols(other.extent(1)) {
          size_t size = m_rows * m_cols;
          m_data.resize(size);
          std::copy(other.data(), other.data()+other.size(), m_data.begin());
        }

      // Assignment operator
      template <typename U> Matrix<T>& operator=(const Matrix<U>& other) {
        m_rows = other.rows();
        m_cols = other.cols();
        m_data.resize(other.size());
        std::copy(other.begin(), other.end(), m_data.begin());
        return *this;
      }

      template <typename U>
        Matrix<T>& operator=(const blitz::Array<U,2>& other) {
          m_rows = other.rows();
          m_cols = other.cols();
          m_data.resize(other.size());
          std::copy(other.data(), other.data()+other.size(), m_data.begin());
          return *this;
        }

      // Equality operator
      template <typename U>
        bool operator==(const Matrix<U>& other) const
        {
          return	m_rows == other.rows() &&
            m_cols == other.cols() &&
            std::equal(begin(), end(), other.begin());
        }
      template <typename U>
        bool operator!=(const Matrix<U>& other) const
        {
          return	!(*this == other);
        }

      // Access functions
      const T* operator[](size_t row) const { return &m_data[row * m_cols]; }

      T* operator[](size_t row) { return &m_data[row * m_cols]; }

      const T& operator()(size_t row, size_t col) const
      { return m_data[row * m_cols + col]; }

      T& operator()(size_t row, size_t col)
      { return m_data[row * m_cols + col]; }

      const T& operator()(size_t index) const { return m_data[index]; }
      T& operator()(size_t index) { return m_data[index]; }

      size_t cols() const { return m_cols; }
      size_t rows() const { return m_rows; }
      size_t size() const { return m_data.size(); }
      bool empty() const { return m_data.empty(); }

      // 1D (row & column) iterators
      ConstDeltaIterator<T> row_begin(size_t row) const
      { return _row_begin(row); }

      DeltaIterator<T> row_begin(size_t row)
      { return _row_begin(row); }

      ConstDeltaIterator<T> row_end(size_t row) const
      { return _row_end(row); }

      DeltaIterator<T> row_end(size_t row) { return _row_end(row); }

      ConstDeltaIterator<T> col_begin(size_t col) const
      { return _col_begin(col); }

      DeltaIterator<T> col_begin(size_t col)
      { return _col_begin(col); }

      ConstDeltaIterator<T> col_end(size_t col) const
      { return _col_end(col); }

      DeltaIterator<T> col_end(size_t col) { return _col_end(col); }

      // 2D (matrix) iterators
      typename std::vector<T>::const_iterator begin() const { return m_data.begin(); }
      typename std::vector<T>::iterator begin() { return m_data.begin(); }

      typename std::vector<T>::const_iterator end() const
      { return m_data.begin() + m_data.size(); }

      typename std::vector<T>::iterator end() { return m_data.begin() + m_data.size(); }

      // Fill
      void fill(T value) { std::fill(begin(), end(), value); }

      // Scale all values
      void scale(T value) {
        std::transform(begin(), end(), begin(), std::bind2nd(std::multiplies<T>(), value));
      }

      // Resize
      void clear() {
        m_cols = m_rows = 0;
        m_data.clear();
      }

      void resize(size_t rows, size_t cols, T fillValue = T()) {
        m_rows = rows, m_cols = cols;
        m_data.resize(m_rows * m_cols, fillValue);
      }

    private:

      // Serialize the object
      friend class boost::serialization::access;
      template <typename Archive>
        void serialize(Archive& ar, const unsigned int) {
          ar & m_rows;
          ar & m_cols;
          ar & m_data;
        }

      // 1D row iterators
      const T* _rpt(size_t row) const { return &m_data[row * m_cols]; }

      T* _rpt(size_t row) { return &m_data[row * m_cols]; }

      ConstDeltaIterator<T> _row_begin(size_t row) const
      { return ConstDeltaIterator<T>(_rpt(row), 1); }

      DeltaIterator<T> _row_begin(size_t row)
      { return DeltaIterator<T>(_rpt(row), 1); }

      ConstDeltaIterator<T> _row_end(size_t row) const
      { return ConstDeltaIterator<T>(_rpt(row) + m_cols, 1); }

      DeltaIterator<T> _row_end(size_t row)
      { return DeltaIterator<T>(_rpt(row) + m_cols, 1); }

      // 1D column iterators
      const T* _cpt(size_t col) const { return &m_data[col]; }
      T* _cpt(size_t col) { return &m_data[col]; }

      ConstDeltaIterator<T> _col_begin(size_t col) const
      { return ConstDeltaIterator<T>(_cpt(col), m_rows); }

      DeltaIterator<T> _col_begin(size_t col)
      { return DeltaIterator<T>(_cpt(col), m_rows); }

      ConstDeltaIterator<T> _col_end(size_t col) const {
        return ConstDeltaIterator<T>(_cpt(col) + (m_data.size() - m_rows), m_rows);
      }

      DeltaIterator<T> _col_end(size_t col) {
        return DeltaIterator<T>(_cpt(col) + (m_data.size() - m_rows), m_rows);
      }

    private:

      // Attributes
      size_t	m_rows, m_cols;		// Size
      std::vector<T> m_data;
  };

}}

#endif // BOB_VISIONER_MATRIX_H
