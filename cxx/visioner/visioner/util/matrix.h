#ifndef BOB_VISIONER_MATRIX_H
#define BOB_VISIONER_MATRIX_H

#include <vector>
#include <boost/serialization/access.hpp>
#include <boost/serialization/vector.hpp>

#include "visioner/util/iterators.h"

namespace bob { namespace visioner {	

  /////////////////////////////////////////////////////////////////////////////////////////
  // 2D matrix using an 1D STL vector.
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename T>
    class Matrix
    {
      public:

        typedef std::vector<T>			data_t;
        typedef typename data_t::const_iterator	d2_const_it_t;
        typedef typename data_t::iterator	d2_it_t;

        typedef DeltaIterator<T>                d1_it_t;
        typedef ConstDeltaIterator<T>           d1_const_it_t;

        // Constructor
        Matrix(std::size_t rows = 0, std::size_t cols = 0, T fillValue = T())
          :	m_rows(rows), m_cols(cols), 
          m_data(rows * cols, fillValue)
      {
      }

        // Starts from an existing pointer
        Matrix(std::size_t rows, std::size_t cols, const T* data)
          : m_rows(rows), m_cols(cols) {
            std::size_t size = m_rows * m_cols;
            m_data.resize(size);
            std::copy(data, data+size, m_data.begin());
          }

        // Assignment operator
        template <typename U>
          Matrix<T>& operator=(const Matrix<U>& other)
          {
            m_rows = other.rows();
            m_cols = other.cols();
            m_data.resize(other.size());
            std::copy(other.begin(), other.end(), m_data.begin());
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
        const T* operator[](std::size_t row) const { return &m_data[row * m_cols]; }
        T* operator[](std::size_t row) { return &m_data[row * m_cols]; }

        const T& operator()(std::size_t row, std::size_t col) const { return m_data[row * m_cols + col]; }
        T& operator()(std::size_t row, std::size_t col) { return m_data[row * m_cols + col]; } 

        const T& operator()(std::size_t index) const { return m_data[index]; }
        T& operator()(std::size_t index) { return m_data[index]; } 

        std::size_t cols() const { return m_cols; }
        std::size_t rows() const { return m_rows; }
        std::size_t size() const { return m_data.size(); }
        bool empty() const { return m_data.empty(); }

        // 1D (row & column) iterators
        d1_const_it_t row_begin(std::size_t row) const { return _row_begin(row); }
        d1_it_t row_begin(std::size_t row) { return _row_begin(row); }                

        d1_const_it_t row_end(std::size_t row) const { return _row_end(row); }
        d1_it_t row_end(std::size_t row) { return _row_end(row); }

        d1_const_it_t col_begin(std::size_t col) const { return _col_begin(col); }
        d1_it_t col_begin(std::size_t col) { return _col_begin(col); }                

        d1_const_it_t col_end(std::size_t col) const { return _col_end(col); }
        d1_it_t col_end(std::size_t col) { return _col_end(col); }

        // 2D (matrix) iterators
        d2_const_it_t begin() const { return m_data.begin(); }
        d2_it_t begin() { return m_data.begin(); }

        d2_const_it_t end() const { return m_data.begin() + m_data.size(); }
        d2_it_t end() { return m_data.begin() + m_data.size(); }               

        // Fill 
        void fill(T value)
        {
          std::fill(begin(), end(), value);
        }

        // Scale all values
        void scale(T value)
        {
          std::transform(begin(), end(), begin(), std::bind2nd(std::multiplies<T>(), value));
        }

        // Resize
        void clear()
        {
          m_cols = m_rows = 0;
          m_data.clear();
        }

        void resize(std::size_t rows, std::size_t cols, T fillValue = T())
        {
          m_rows = rows, m_cols = cols;							
          m_data.resize(m_rows * m_cols, fillValue);
        }	

      private:

        // Serialize the object
        friend class boost::serialization::access;
        template <typename Archive>
          void serialize(Archive& ar, const unsigned int)
          {
            ar & m_rows;
            ar & m_cols;
            ar & m_data;
          }

        // 1D row iterators
        const T* _rpt(std::size_t row) const { return &m_data[row * m_cols]; }
        T* _rpt(std::size_t row) { return &m_data[row * m_cols]; }

        d1_const_it_t _row_begin(std::size_t row) const { return d1_const_it_t(_rpt(row), 1); }
        d1_it_t _row_begin(std::size_t row) { return d1_it_t(_rpt(row), 1); }

        d1_const_it_t _row_end(std::size_t row) const { return d1_const_it_t(_rpt(row) + m_cols, 1); }
        d1_it_t _row_end(std::size_t row) { return d1_it_t(_rpt(row) + m_cols, 1); }

        // 1D column iterators
        const T* _cpt(std::size_t col) const { return &m_data[col]; }
        T* _cpt(std::size_t col) { return &m_data[col]; }

        d1_const_it_t _col_begin(std::size_t col) const { return d1_const_it_t(_cpt(col), m_rows); }
        d1_it_t _col_begin(std::size_t col) { return d1_it_t(_cpt(col), m_rows); }

        d1_const_it_t _col_end(std::size_t col) const 
        { 
          return d1_const_it_t(_cpt(col) + (m_data.size() - m_rows), m_rows); 
        }
        d1_it_t _col_end(std::size_t col) 
        { 
          return d1_it_t(_cpt(col) + (m_data.size() - m_rows), m_rows); 
        }

      private:

        // Attributes
        std::size_t		m_rows, m_cols;		// Size
        data_t			m_data;
    };

}}

#endif // BOB_VISIONER_MATRIX_H
