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

namespace Torch {
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

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
        bool needCast(const blitz::Array<T,D>& bl) const;
        /************** Partial specialization declaration *************/
        template<int D> bool needCast(const blitz::Array<bool,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<int8_t,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<int16_t,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<int32_t,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<int64_t,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<uint8_t,D>& bl) const;
        template<int D> 
        bool needCast(const blitz::Array<uint16_t,D>& bl) const;
        template<int D> 
        bool needCast(const blitz::Array<uint32_t,D>& bl) const;
        template<int D> 
        bool needCast(const blitz::Array<uint64_t,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<float,D>& bl) const;
        template<int D> bool needCast(const blitz::Array<double,D>& bl) const;
        template<int D> 
        bool needCast(const blitz::Array<std::complex<float>,D>& bl) const;
        template<int D>
        bool needCast(const blitz::Array<std::complex<double>,D>& bl) const;

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

    template <typename T, int d>
    bool BinFileHeader::needCast(const blitz::Array<T,d>& bl) const
    {
      error << "Unsupported blitz array type " << std::endl;
      throw Exception();
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<bool,d>& bl) const
    {
      if(m_elem_type == array::t_bool )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<int8_t,d>& bl) const
    {
      if(m_elem_type == array::t_int8 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<int16_t,d>& bl) const
    {
      if(m_elem_type == array::t_int16 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<int32_t,d>& bl) const
    {
      if(m_elem_type == array::t_int32 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<int64_t,d>& bl) const
    {
      if(m_elem_type == array::t_int64 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<uint8_t,d>& bl) const
    {
      if(m_elem_type == array::t_uint8 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<uint16_t,d>& bl) const
    {
      if(m_elem_type == array::t_uint16 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<uint32_t,d>& bl) const
    {
      if(m_elem_type == array::t_uint32 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<uint64_t,d>& bl) const
    {
      if(m_elem_type == array::t_uint64 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<float,d>& bl) const
    {
      if(m_elem_type == array::t_float32 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(const blitz::Array<double,d>& bl) const
    {
      if(m_elem_type == array::t_float64 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(
      const blitz::Array<std::complex<float>,d>& bl) const
    {
      if(m_elem_type == array::t_complex64 )
        return false;
      return true;
    }

    template <int d>
    bool BinFileHeader::needCast(
      const blitz::Array<std::complex<double>,d>& bl) const
    {
      if(m_elem_type == array::t_complex128 )
        return false;
      return true;
    }

  }
}


#endif /* TORCH5SPRO_CORE_BIN_FILE_HEADER_H */

