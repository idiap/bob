/**
 * @file src/core/core/BinInputFile.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to load multiarrays from binary files.
 */

#ifndef TORCH5SPRO_CORE_BIN_INPUT_FILE_H
#define TORCH5SPRO_CORE_BIN_INPUT_FILE_H

#include "core/BinFileHeader.h"
#include "core/Dataset2.h"
#include "core/StaticComplexCast.h"

namespace Torch {
  namespace core {

    /**
     *  @brief the InputFile class for loading multiarrays from binary files
     */
    class BinInputFile
    {
      public:
        /**
         * @brief Constructor
         */
        BinInputFile(const std::string& filename);

        /**
         * @brief Destructor
         */
        ~BinInputFile();

        /**
         * Close the BlitzInputFile
         */
        void    close();

        //TODO: Make the API more consistent
        /**
         * Load one blitz++ multiarray from the input stream/file
         * All the multiarrays saved have the same dimensions
         */
        template <typename T, int d> void load( blitz::Array<T,d>& bl);
        template <typename T, int d> 
        void load(size_t index, blitz::Array<T,d>& bl);

        /**
         * @brief Get one C-style array from the input stream/file, and cast
         * it to the given type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions
         */
        template <typename T> BinInputFile& operator>>(T* multiarray);

        /** 
         * @brief Load an Arrayset from a binary file
         */
        void load( Arrayset& arrayset);

        /** 
         * @brief Load an Array from a binary file
         */
        void load( Array& array);


        /**
         * Return the header
         */
        BinFileHeader& getHeader() { return m_header; }

      private:
        size_t m_current_array;
        std::fstream m_in_stream;
        BinFileHeader m_header;
    };


    template <typename T, int d> 
    void BinInputFile::load( blitz::Array<T,d>& bl) {
      // Check the shape compatibility (number of dimensions)
      if( d != m_header.getNDimensions() ) {
        error << "The dimensions of this array does not match the " <<
          "ones contained in the header file. The array cannot be loaded." <<
          std::endl;
        throw Exception();
      }

      // Reshape each dimension with the correct size
      blitz::TinyVector<int,d> shape;
      m_header.getShape(shape);
      bl.resize(shape);
     
      // Check that the memory of the blitz array is contiguous (maybe useless)
      // TODO: access the data in an other way and do not raise an exception
      if( !bl.isStorageContiguous() ) {
        error << "The memory of the blitz array is not contiguous." <<
          "The array cannot be loaded." <<
          std::endl;
        throw Exception();
      }
        
      T* data = bl.data();
      // copy the data from the input stream to the blitz array
      operator>>(data);
    }

    template <typename T, int d> 
    void BinInputFile::load( size_t index, blitz::Array<T,d>& bl) {
      // Set the stream pointer at the correct position
      m_in_stream.seekg( m_header.getArrayIndex(index) );

      // Put the content of the stream in the blitz array.
      load(bl);
    }

    template <typename T> BinInputFile& BinInputFile::operator>>(T* multiarray) {
      // copy the multiarray from the input stream to the C-style array
      bool b;
      int8_t i8; int16_t i16; int32_t i32; int64_t i64;
      uint8_t ui8; uint16_t ui16; uint32_t ui32; uint64_t ui64;
      float f; double dou; long double ld;
      std::complex<float> cf; std::complex<double> cd; 
      std::complex<long double> cld;
      switch(m_header.getArrayType())
      {
        case array::t_bool:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> b;
            static_complex_cast(b,multiarray[i]);
          }
          break;
        case array::t_int8:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> i8;
            static_complex_cast(i8,multiarray[i]);
          }
          break;
        case array::t_int16:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> i16;
            static_complex_cast(i16,multiarray[i]);
          }
          break;
        case array::t_int32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> i32;
            static_complex_cast(i32,multiarray[i]);
          }
          break;
        case array::t_int64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> i64;
            static_complex_cast(i64,multiarray[i]);
          }
          break;
        case array::t_uint8:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> ui8;
            static_complex_cast(ui8,multiarray[i]);
          }
          break;
        case array::t_uint16:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> ui16;
            static_complex_cast(ui16,multiarray[i]);
          }
          break;
        case array::t_uint32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> ui32;
            static_complex_cast(ui32,multiarray[i]);
          }
          break;
        case array::t_uint64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> ui64;
            static_complex_cast(ui64,multiarray[i]);
          }
          break;
        case array::t_float32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> f;
            static_complex_cast(f,multiarray[i]);
          }
          break;
        case array::t_float64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> dou;
            static_complex_cast(dou,multiarray[i]);
          }
          break;
        case array::t_float128:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> ld;
            static_complex_cast(ld,multiarray[i]);
          }
          break;
        case array::t_complex64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> cf;
            static_complex_cast(cf,multiarray[i]);
          }
          break;
        case array::t_complex128:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> cd;
            static_complex_cast(cd,multiarray[i]);
          }
          break;
        case array::t_complex256:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            m_in_stream >> cld;
            static_complex_cast(cld,multiarray[i]);
          }
          break;
        default:
          break;
      }

      return *this;
    }

  }
}

#endif /* TORCH5SPRO_CORE_BIN_INPUT_FILE_H */

