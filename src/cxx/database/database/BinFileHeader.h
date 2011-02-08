/**
 * @file src/cxx/database/database/BinFileHeader.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class defines an header for storing multiarrays into
 * binary files.
 */

#ifndef TORCH_DATABASE_BINFILEHEADER_H
#define TORCH_DATABASE_BINFILEHEADER_H

#include <fstream>
#include <blitz/array.h>

namespace Torch { namespace database { namespace detail {

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
      inline size_t getNElements() { 
        size_t tmp = 1;
        for(size_t i=0; i<m_n_dimensions; ++i) tmp *= m_shape[i];
      }

      //representation
      uint8_t m_version; ///< current version being read
      Torch::core::array::ElementType m_elem_type; ///< array element type 
      uint8_t m_elem_sizeof; ///< the syze in bytes of the element
      uint32_t m_endianness; ///< the endianness of data recorded in the file
      uint8_t m_n_dimensions; ///< the number of dimensions in each array
      size_t m_shape[array::N_MAX_DIMENSIONS_ARRAY]; ///< shape of data
      uint64_t m_n_samples; ///< total number of arrays in the file
      uint64_t m_n_elements; ///< number of elements per array == PROD(shape)
    };

} } }

#endif /* TORCH_DATABASE_BINFILEHEADER_H */

