/**
 * @file src/core/core/BinOutputFile.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store multiarrays into files.
 */

#ifndef TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H
#define TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H

#include "core/BinFileHeader.h"
#include "core/Dataset2.h"
#include "core/StaticComplexCast.h"
#include <string>

namespace Torch {
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

    /**
     *  @brief the OutputFile class for storing multiarrays into files
     */
    class BinOutputFile
    {
      public:
        /**
         * @brief Constructor
         */
        BinOutputFile(const std::string& filename, bool append = false);

        /**
         * @brief Destructor
         */
        virtual ~BinOutputFile();

        /**
         * @brief Initialize the header of the output stream with the given
         * type and shape
         */
        void initHeader(const array::ArrayType type, 
            const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]);

        /**
         * @brief Close the BlitzOutputFile and update the header
         */
        void close();


        // TODO: Make API more consistent?
        /**
         * @brief Put a void C-style multiarray into the output stream/file
         * @warning This is the responsability of the user to check
         * the correctness of the type and size of the memory block 
         * pointed by the void pointer
         */
        BinOutputFile& operator<<(const void* multi_array);

        /** 
         * @brief Put a C-style multiarray of a given type into the output
         * stream/file by casting it to the correct type.
         */
        template <typename T> BinOutputFile& operator<<(const T* multi_array);

        /** 
         * @brief Put a Blitz++ multiarray of a given type into the output
         * stream/file by casting it to the correct type.
         */
        template <typename T, int D> void save(const blitz::Array<T,D>& bl);

        /**
         * @brief Save an Arrayset into a binary file
         */
        void save(const Arrayset& arrayset);

        /**
         * @brief Save an Array into a binary file
         */
        void save(const Array& array);

        /**
         * @brief Return the header
         */
        BinFileHeader& getHeader() { return m_header; }

      private:
        /**
         * @brief Check that the header is initialized (before writing data)
         */
        void checkHeaderInit();

        bool m_header_init;
        std::fstream m_out_stream;
        BinFileHeader m_header;
        size_t m_n_arrays_written;
    };

    template <typename T> BinOutputFile& BinOutputFile::operator<<(
      const T* multi_array) 
    {
      // Check that the header has been initialized
      checkHeaderInit();

      // copy the data into the output stream
      bool b;
      int8_t i8; int16_t i16; int32_t i32; int64_t i64;
      uint8_t ui8; uint16_t ui16; uint32_t ui32; uint64_t ui64;
      float f; double d; long double ld;
      std::complex<float> cf; std::complex<double> cd; 
      std::complex<long double> cld;
      switch(m_header.getArrayType())
      {
        case array::t_bool:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],b);
            m_out_stream << b;
          }
          break;
        case array::t_int8:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i8);
            m_out_stream << i8;
          }
          break;
        case array::t_int16:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i16);
            m_out_stream << i16;
          }
          break;
        case array::t_int32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i32);
            m_out_stream << i32;
          }
          break;
        case array::t_int64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i64);
            m_out_stream << i64;
          }
          break;
        case array::t_uint8:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui8);
            m_out_stream << ui8;
          }
          break;
        case array::t_uint16:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui16);
            m_out_stream << ui16;
          }
          break;
        case array::t_uint32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui32);
            m_out_stream << ui32;
          }
          break;
        case array::t_uint64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui64);
            m_out_stream << ui64;
          }
          break;
        case array::t_float32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],f);
            m_out_stream << f;
          }
          break;
        case array::t_float64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],d);
            m_out_stream << d;
          }
          break;
        case array::t_float128:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ld);
            m_out_stream << ld;
          }
          break;
        case array::t_complex64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],cf);
            m_out_stream << cf;
          }
          break;
        case array::t_complex128:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],cd);
            m_out_stream << cd;
          }
          break;
        case array::t_complex256:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],cld);
            m_out_stream << cld;
          }
          break;
        default:
          break;
      }

      // increment m_n_arrays_written
      ++m_n_arrays_written;

      return *this;
    }


    template <typename T, int D> 
    void BinOutputFile::save(const blitz::Array<T,D>& bl) {
      // Check that the header has been initialized
      checkHeaderInit();

      // Check the shape compatibility
      bool shapeCompatibility = true;
      size_t i=0;
      const size_t* h_shape = m_header.getShape();
      while( i<array::N_MAX_DIMENSIONS_ARRAY && shapeCompatibility) {
        shapeCompatibility = (bl.extent(i) == h_shape[i]);
        ++i;
      }

      if(!shapeCompatibility)
      {
        error << "The dimensions of this array does not match the " <<
          "contained in the header file. The array cannot be saved." <<
          std::endl;
        throw Exception();
      }

      // Copy the data into the output stream
      const T* data;
      if( bl.isStorageContiguous() )
        data = bl.data();
      else
        data = bl.copy().data();
      operator<<(data);
    }

  }
}

#endif /* TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H */

