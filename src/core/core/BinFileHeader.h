/**
 * @file src/core/core/BinFileHeader.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class defines an header for storing multiarrays into
 * binary files.
 */

#ifndef TORCH5SPRO_CORE_BIN_FILE_HEADER_H
#define TORCH5SPRO_CORE_BIN_FILE_HEADER_H

#include "core/logging.h"
#include "core/Exception.h"
#include <fstream>
#include <blitz/array.h>

namespace Torch {
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

    /**
     *  @brief the Header class for storing multiarrays into binary files
     */
    class BinFileHeader
    {
      public:
        /**
         * @brief Constructor
         */
        BinFileHeader();

        /**
         * @brief Destructor
         */
        virtual ~BinFileHeader() {}

        /**
         * @brief Get the version
         */
        size_t getVersion() const { return m_version; }
        /**
         * @brief Get the Array type
         */
        array::ArrayType getArrayType() const { return m_type; }
        /**
         * @brief Get the number of dimensions
         */
        size_t getNDimensions() const { return m_n_dimensions; }
        /**
         * @brief Get the shape of each array
         */
        const size_t* getShape() const { return m_shape; }
        /**
         * @brief Get the number of samples/arrays
         */
        size_t getNSamples() const { return m_n_samples; }
        /**
         * @brief Get the Endianness
         */
        size_t getEndianness() const { return m_endianness; }
        /**
         * @brief Get the number of elements per array
         */
        size_t getNElements() const { return m_n_elements; }
        /**
         * @brief Get the size of the type of the array elements
         */
        size_t getDataSizeof() const { return m_data_sizeof; }

        /**
         * @brief Set the version
         */
        void setVersion(const size_t version) { m_version = version; }
        /**
         * @brief Set the Array type
         */
        void setArrayType(const array::ArrayType type) { m_type = type; }
        /**
         * @brief Set the number of dimensions
         */
        void setNDimensions(const size_t n_dim) { m_n_dimensions = n_dim; }
        /**
         * @brief Set the shape of each array
         */
        void setShape(const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]) {
          for(size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
            m_shape[i] = shape[i];
          sizeUpdated();
        }
        /**
         * @brief Set the number of samples
         */
        void setNSamples(const size_t n_samples) { m_n_samples = n_samples; }
        /**
         * @brief Set the Endianness
         */
        void setEndianness(const size_t endianness) { 
          m_endianness = endianness; }


        /**
         * @brief Get the size along a particular dimension
         */
        size_t getSize(size_t dim_index) const;
        /**
         * @brief Set the size along a particular dimension
         */
        void setSize(const size_t dim_index, size_t val);
        /** 
         * @brief Get the offset of some array in the file
         */
        size_t getArrayIndex(size_t index) const;

        /**
         * @brief Write the header into an output stream
         */
        void write(std::ostream& str) const;
        /**
         * @brief Read the header from an input stream
         */
        void read(std::istream& str);


      private:
        /**
         * @brief Update the number of elements and number of dimensions 
         * members (is called in case of resizing)
         */
        void sizeUpdated();
        /**
         * @brief Update the size of the array elements' type
         * (is called when the type is set)
         */
        void typeUpdated();

        /**
         *  Attributes
         */
        size_t m_version;
        array::ArrayType m_type;
        size_t m_n_dimensions;
        size_t m_shape[array::N_MAX_DIMENSIONS_ARRAY];
        size_t m_n_samples;
        size_t m_endianness;
        size_t m_n_elements;
        size_t m_data_sizeof;
    };

  }
}


#endif /* TORCH5SPRO_CORE_BIN_FILE_HEADER_H */

