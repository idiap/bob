/**
 * @file src/cxx/core/core/BinFileHeader.h
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

#include <typeinfo>

namespace Torch {
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

    namespace BinaryFile {
      extern const uint32_t MAGIC_ENDIAN_DW;
      extern const uint8_t FORMAT_VERSION;
    }

    /**
     *  @brief The Header for storing multiarrays into binary files
     */
    struct BinFileHeader
    {
        /**
         * @brief Constructor
         */
        BinFileHeader();

        /**
         * @brief Destructor
         */
        virtual ~BinFileHeader() {}

        /**
         * @brief Get the shape of each array in a blitz format
         */
        template<int d>
        void getShape( blitz::TinyVector<int,d>& res ) const {
          for( int i=0; i<d; ++i)
            res[i] = m_shape[i];
        }

        /**
         * @brief Set the shape of each array
         */
        void setShape(const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]) {
          for(size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i)
            m_shape[i] = shape[i];
          sizeUpdated();
        }

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

        /**
         * @brief Check if there is a need to cast the data of a blitz array
         */
        template <typename T, int D> 
        bool needCast(const blitz::Array<T,D>& bl) const {
          std::cout << "Generic: " << typeid(T).name() << std::endl;
          return true;
        }

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
        uint8_t m_version;
        array::ElementType m_elem_type;
        uint8_t m_elem_sizeof;
        uint8_t m_n_dimensions;
        uint32_t m_endianness;
        size_t m_shape[array::N_MAX_DIMENSIONS_ARRAY];
        uint64_t m_n_samples;

        uint64_t m_n_elements;
    };


/************** Full specialization declarations *************/
#define NEED_CAST_DECL(T,D) template<> bool \
  BinFileHeader::needCast(const blitz::Array<T,D>& bl) const;\

    NEED_CAST_DECL(bool,1)
    NEED_CAST_DECL(bool,2)
    NEED_CAST_DECL(bool,3)
    NEED_CAST_DECL(bool,4)
    NEED_CAST_DECL(int8_t,1)
    NEED_CAST_DECL(int8_t,2)
    NEED_CAST_DECL(int8_t,3)
    NEED_CAST_DECL(int8_t,4)
    NEED_CAST_DECL(int16_t,1)
    NEED_CAST_DECL(int16_t,2)
    NEED_CAST_DECL(int16_t,3)
    NEED_CAST_DECL(int16_t,4)
    NEED_CAST_DECL(int32_t,1)
    NEED_CAST_DECL(int32_t,2)
    NEED_CAST_DECL(int32_t,3)
    NEED_CAST_DECL(int32_t,4)
    NEED_CAST_DECL(int64_t,1)
    NEED_CAST_DECL(int64_t,2)
    NEED_CAST_DECL(int64_t,3)
    NEED_CAST_DECL(int64_t,4)
    NEED_CAST_DECL(uint8_t,1)
    NEED_CAST_DECL(uint8_t,2)
    NEED_CAST_DECL(uint8_t,3)
    NEED_CAST_DECL(uint8_t,4)
    NEED_CAST_DECL(uint16_t,1)
    NEED_CAST_DECL(uint16_t,2)
    NEED_CAST_DECL(uint16_t,3)
    NEED_CAST_DECL(uint16_t,4)
    NEED_CAST_DECL(uint32_t,1)
    NEED_CAST_DECL(uint32_t,2)
    NEED_CAST_DECL(uint32_t,3)
    NEED_CAST_DECL(uint32_t,4)
    NEED_CAST_DECL(uint64_t,1)
    NEED_CAST_DECL(uint64_t,2)
    NEED_CAST_DECL(uint64_t,3)
    NEED_CAST_DECL(uint64_t,4)
    NEED_CAST_DECL(float,1)
    NEED_CAST_DECL(float,2)
    NEED_CAST_DECL(float,3)
    NEED_CAST_DECL(float,4)
    NEED_CAST_DECL(double,1)
    NEED_CAST_DECL(double,2)
    NEED_CAST_DECL(double,3)
    NEED_CAST_DECL(double,4)
    NEED_CAST_DECL(std::complex<float>,1)
    NEED_CAST_DECL(std::complex<float>,2)
    NEED_CAST_DECL(std::complex<float>,3)
    NEED_CAST_DECL(std::complex<float>,4)
    NEED_CAST_DECL(std::complex<double>,1)
    NEED_CAST_DECL(std::complex<double>,2)
    NEED_CAST_DECL(std::complex<double>,3)
    NEED_CAST_DECL(std::complex<double>,4)

  }
}


#endif /* TORCH5SPRO_CORE_BIN_FILE_HEADER_H */

