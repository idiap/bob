/**
 * @file cxx/io/io/BinFileHeader.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This class defines an header for storing multiarrays into
 * binary files.
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

#ifndef BOB_IO_BINFILEHEADER_H
#define BOB_IO_BINFILEHEADER_H

#include <fstream>
#include <blitz/array.h>
#include "core/array_type.h"

namespace bob { namespace io { namespace detail {

    extern const uint32_t MAGIC_ENDIAN_DW;
    extern const uint8_t FORMAT_VERSION;

    /**
     * The Header for storing arrays into binary files. Please note that this
     * class is for private use of the BinFile type.
     */
    struct BinFileHeader {

      /**
       * Constructor
       */
      BinFileHeader();

      /**
       * Destructor
       */
      virtual ~BinFileHeader();

      /**
       * Gets the shape of each array in a blitz format
       */
      template<int D> void getShape (blitz::TinyVector<int,D>& res) const {
        for (int i=0; i<D; ++i) res[i] = m_shape[i];
      }

      /**
       * Sets the shape of each array
       */
      void setShape(size_t ndim, const size_t* shape) {
        m_n_dimensions = ndim;
        for(size_t i=0; i<ndim; ++i) m_shape[i] = shape[i];
      }

      /**
       * Gets the size along a particular dimension
       */
      size_t getSize(size_t dim_index) const;

      /**
       * Sets the size along a particular dimension
       */
      void setSize(const size_t dim_index, size_t val);

      /** 
       * Gets the offset of some array in the file
       */
      size_t getArrayIndex(size_t index) const;

      /**
       * Writes the header into an output stream
       */
      void write(std::ostream& str) const;

      /**
       * Reads the header from an input stream
       */
      void read(std::istream& str);

      /**
       * Gets number of elements in binary file
       */
      inline size_t getNElements() const {
        size_t tmp = 1;
        for(size_t i=0; i<m_n_dimensions; ++i) tmp *= m_shape[i];
        return tmp;
      }

      /**
       * Returns the number of dimensions in this binary file
       */
      inline size_t getNDim() const { return m_n_dimensions; }

      /**
       * Returns the shape in a N-element C-style array
       */
      inline const size_t* getShape() const { return m_shape; }

      //representation
      uint8_t m_version; ///< current version being read
      bob::core::array::ElementType m_elem_type; ///< array element type 
      uint8_t m_elem_sizeof; ///< the syze in bytes of the element
      uint32_t m_endianness; ///< the endianness of data recorded in the file
      uint8_t m_n_dimensions; ///< the number of dimensions in each array
      size_t m_shape[bob::core::array::N_MAX_DIMENSIONS_ARRAY]; ///< shape of data
      uint64_t m_n_samples; ///< total number of arrays in the file
      uint64_t m_n_elements; ///< number of elements per array == PROD(shape)
    };

} } }

#endif /* BOB_IO_BINFILEHEADER_H */

