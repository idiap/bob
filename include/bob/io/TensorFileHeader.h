/**
 * @file cxx/io/io/TensorFileHeader.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This class defines an header for storing multiarrays into
 * .tensor files.
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

#ifndef BOB_IO_TENSORFILEHEADER_H
#define BOB_IO_TENSORFILEHEADER_H

#include <fstream>
#include <blitz/array.h>
#include "bob/core/array.h"

namespace bob { namespace io { 

  // TensorType
  enum TensorType
  {
    Char,
    Short,
    Int,
    Long,
    Float,
    Double
  };

  TensorType arrayTypeToTensorType(bob::core::array::ElementType eltype);
  bob::core::array::ElementType tensorTypeToArrayType(bob::io::TensorType tensortype);

  namespace detail {
    /**
     * The Header for storing arrays into binary files. Please note that this
     * class is for private use of the TensorFile type.
     */
    struct TensorFileHeader {

      /**
       * Constructor
       */
      TensorFileHeader();

      /**
       * Destructor
       */
      virtual ~TensorFileHeader();

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
        for(size_t i=0; i<m_type.nd; ++i) tmp *= m_type.shape[i];
        return tmp;
      }

      /**
        * Checks if the header is valid
        */
      void header_ok();

      /**
        * Update the TensorSize value
        */
      void update();

      //representation
      TensorType m_tensor_type; ///< array element type 
      bob::core::array::typeinfo m_type; ///< the type information
      size_t m_n_samples; ///< total number of arrays in the file
      size_t m_tensor_size; ///< the number of dimensions in each array
    };

} } }

#endif /* BOB_IO_TENSORFILEHEADER_H */
