/**
 * @file src/cxx/core/core/BinInputFile.h
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
     *  @brief The InputFile class for loading multiarrays from binary files
     *  @deprecated Please use the BinFile class instead.
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
         * Close the BinInputFile
         */
        void close();

        /**
         * @brief Load one blitz++ multiarray from the input stream/file
         * All the multiarrays saved have the same dimensions.
         */
        template <typename T, int d> void read( blitz::Array<T,d>& bl);
        template <typename T, int d> 
        void read(size_t index, blitz::Array<T,d>& bl);

        /** 
         * @brief Load an Arrayset from a binary file
         */
        void read( Arrayset& arrayset);

        /** 
         * @brief Load an Array from a binary file
         */
        void read( Array& array);


        /**
         * @brief Get the Element type
         */
        array::ElementType getElementType() const { 
          return m_header.m_elem_type; 
        }
        /**
         * @brief Get the number of dimensions
         */
        size_t getNDimensions() const { return m_header.m_n_dimensions; }
        /**
         * @brief Get the shape of each array
         */
        const size_t* getShape() const { return m_header.m_shape; }
        /**
         * @brief Get the shape of each array in a blitz format
         */
        template<int d>
        void getShape( blitz::TinyVector<int,d>& res ) const {
          m_header.getShape(res);
        }
        /**
         * @brief Get the number of samples/arrays written so far
         */
        size_t getNSamples() const { return m_header.m_n_samples; }
        /**
         * @brief Get the number of elements per array
         */
        size_t getNElements() const { return m_header.m_n_elements; }
        /**
         * @brief Get the size along a particular dimension
         */
        size_t getSize(size_t dim_index) const { 
          return m_header.getSize(dim_index); 
        }


      private:
        /**
         * @brief Put a void C-style multiarray into the output stream/file
         * @warning This is the responsability of the user to check
         * the correctness of the type and size of the memory block 
         * pointed by the void pointer
         */
        BinInputFile& read(void* multi_array);

        /**
         * @brief Get one C-style array from the input stream/file, and cast
         * it to the given type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions
         */
        template <typename T> BinInputFile& readWithCast(T* multiarray);

        /**
         * @brief Check if the end of the binary file is reached
         */
        void endOfFile() {
          if(m_current_array >= m_header.m_n_samples ) {
            error << "The end of the binary file has been reached." << 
              std::endl;
            throw Exception();
          }
        }

        /**
         * Attributes
         */
        size_t m_current_array;
        std::fstream m_in_stream;
        BinFileHeader m_header;
    };


    template <typename T, int d> 
    void BinInputFile::read( blitz::Array<T,d>& bl) {
      // Check that the last array was not reached in the binary file
      endOfFile(); 

      // Check the shape compatibility (number of dimensions)
      if( d != m_header.m_n_dimensions ) {
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
          "The array cannot be loaded." << std::endl;
        throw Exception();
      }
        
      T* data = bl.data();
      // copy the data from the input stream to the blitz array
      if( m_header.needCast(bl))
        readWithCast(data);
      else
        read(data);
    }

    template <typename T, int d> 
    void BinInputFile::read( size_t index, blitz::Array<T,d>& bl) {
      // Check that we are reaching an existing array
      if( index > m_header.m_n_samples ) {
        error << "Trying to reach a non-existing array." << std::endl;
        throw Exception();
      }

      // Set the stream pointer at the correct position
      m_in_stream.seekg( m_header.getArrayIndex(index) );
      m_current_array = index;

      // Put the content of the stream in the blitz array.
      read(bl);
    }

    template <typename T> 
    BinInputFile& BinInputFile::readWithCast(T* multiarray) {
      // copy the multiarray from the input stream to the C-style array
      bool b;
      int8_t i8; int16_t i16; int32_t i32; int64_t i64;
      uint8_t ui8; uint16_t ui16; uint32_t ui32; uint64_t ui64;
      float f; double dou;
      std::complex<float> cf; std::complex<double> cd; 
      switch(m_header.m_elem_type)
      {
        case array::t_bool:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&b), sizeof(bool));
            static_complex_cast(b,multiarray[i]);
          }
          break;
        case array::t_int8:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&i8), sizeof(int8_t));
            static_complex_cast(i8,multiarray[i]);
          }
          break;
        case array::t_int16:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&i16), sizeof(int16_t));
            static_complex_cast(i16,multiarray[i]);
          }
          break;
        case array::t_int32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&i32), sizeof(int32_t));
            static_complex_cast(i32,multiarray[i]);
          }
          break;
        case array::t_int64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&i64), sizeof(int64_t));
            static_complex_cast(i64,multiarray[i]);
          }
          break;
        case array::t_uint8:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&ui8), sizeof(uint8_t));
            static_complex_cast(ui8,multiarray[i]);
          }
          break;
        case array::t_uint16:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&ui16), 
              sizeof(uint16_t));
            static_complex_cast(ui16,multiarray[i]);
          }
          break;
        case array::t_uint32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&ui32), 
              sizeof(uint32_t));
            static_complex_cast(ui32,multiarray[i]);
          }
          break;
        case array::t_uint64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&ui64), 
              sizeof(uint64_t));
            static_complex_cast(ui64,multiarray[i]);
          }
          break;
        case array::t_float32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&f), sizeof(float));
            static_complex_cast(f,multiarray[i]);
          }
          break;
        case array::t_float64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&dou), sizeof(double));
            static_complex_cast(dou,multiarray[i]);
          }
          break;
        case array::t_complex64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&cf), 
              sizeof(std::complex<float>));
            static_complex_cast(cf,multiarray[i]);
          }
          break;
        case array::t_complex128:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_in_stream.read( reinterpret_cast<char*>(&cd), 
              sizeof(std::complex<double>));
            static_complex_cast(cd,multiarray[i]);
          }
          break;
        default:
          break;
      }

      // Update current array
      ++m_current_array;

      return *this;
    }

  }
}

#endif /* TORCH5SPRO_CORE_BIN_INPUT_FILE_H */

