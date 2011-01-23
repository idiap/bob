/**
 * @file src/cxx/core/core/BinOutputFile.h
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
     *  @brief The OutputFile class for storing multiarrays into files
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
         * @brief Close the BinOutputFile and update the header
         */
        void close();

        /** 
         * @brief Put a Blitz++ multiarray of a given type into the output
         * stream/file by casting it to the correct type.
         */
        template <typename T, int D> void write(const blitz::Array<T,D>& bl);

        /**
         * @brief Save an Arrayset into a binary file
         */
        void write(const Arrayset& arrayset);

        /**
         * @brief Save an Array into a binary file
         */
        void write(const Array& array);


        /**
         * @brief Get the Array type
         * @warning An exception is thrown if nothing was written so far
         */
        array::ArrayType getArrayType() const { 
          headerInitialized(); 
          return m_header.getArrayType(); 
        }
        /**
         * @brief Get the number of dimensions
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNDimensions() const {  
          headerInitialized(); 
          return m_header.getNDimensions(); 
        }
        /**
         * @brief Get the shape of each array
         * @warning An exception is thrown if nothing was written so far
         */
        const size_t* getShape() const { 
          headerInitialized(); 
          return m_header.getShape(); 
        }
        /**
         * @brief Get the shape of each array in a blitz format
         * @warning An exception is thrown if nothing was written so far
         */
        template<int d>
        void getShape( blitz::TinyVector<int,d>& res ) const {
          headerInitialized(); 
          m_header.getShape(res);
        }
        /**
         * @brief Get the number of samples/arrays written so far
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNSamples() const { 
          headerInitialized(); 
          return m_n_arrays_written; 
        }
        /**
         * @brief Get the number of elements per array
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNElements() const { 
          headerInitialized(); 
          return m_header.getNElements(); 
        }
        /**
         * @brief Get the size along a particular dimension
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getSize(size_t dim_index) const { 
          headerInitialized(); 
          return m_header.getSize(dim_index); 
        }


      private:
        /**
         * @brief Put a void C-style multiarray into the output stream/file
         * @warning This is the responsability of the user to check
         * the correctness of the type and size of the memory block 
         * pointed by the void pointer.
         */
        BinOutputFile& write(const void* multi_array);

        /** 
         * @brief Put a C-style multiarray of a given type into the output
         * stream/file by casting it to the correct type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions.
         */
        template <typename T> 
        BinOutputFile& writeWithCast(const T* multi_array);

        /**
         * @brief Check that the header has been initialized, and raise an
         * exception if not
         */
        void headerInitialized() const { 
          if(!m_header_init) {
            error << "The error has not yet been initialized." << std::endl;
            throw Exception();
          }
        }

        /**
         * @brief Initialize the header of the output stream with the given
         * type and shape
         */
        void initHeader(const array::ArrayType type, 
            const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]);

        /**
         * @brief Initialize the header with a blitz array
         */
        template <typename T, int D> 
        void initHeader(const blitz::Array<T,D>& bl);

        /**
         * @brief Initialize the part of the header which requires 
         * specialization with a blitz array
         */
        template <typename T, int D> 
        void initTypeHeader(const blitz::Array<T,D>& bl);
        /************** Partial specialization declaration *************/
        template<int D> void initTypeHeader(const blitz::Array<bool,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<int8_t,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<int16_t,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<int32_t,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<int64_t,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<uint8_t,D>& bl);
        template<int D> 
        void initTypeHeader(const blitz::Array<uint16_t,D>& bl);
        template<int D> 
        void initTypeHeader(const blitz::Array<uint32_t,D>& bl);
        template<int D> 
        void initTypeHeader(const blitz::Array<uint64_t,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<float,D>& bl);
        template<int D> void initTypeHeader(const blitz::Array<double,D>& bl);
        template<int D> 
        void initTypeHeader(const blitz::Array<long double,D>& bl);
        template<int D> 
        void initTypeHeader(const blitz::Array<std::complex<float>,D>& bl);
        template<int D>
        void initTypeHeader(const blitz::Array<std::complex<double>,D>& bl);
        template<int D> 
        void 
        initTypeHeader(const blitz::Array<std::complex<long double>,D>& bl);

        /**
         * Attributes
         */
        bool m_header_init;
        std::fstream m_out_stream;
        BinFileHeader m_header;
        size_t m_n_arrays_written;
    };


    template <typename T> BinOutputFile& BinOutputFile::writeWithCast(
      const T* multi_array) 
    {
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
            m_out_stream.write( reinterpret_cast<const char*>(&b), 
              sizeof(bool));
          }
          break;
        case array::t_int8:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i8);
            m_out_stream.write( reinterpret_cast<const char*>(&i8), 
              sizeof(int8_t));
          }
          break;
        case array::t_int16:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i16);
            m_out_stream.write( reinterpret_cast<const char*>(&i16), 
              sizeof(int16_t));
          }
          break;
        case array::t_int32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i32);
            m_out_stream.write( reinterpret_cast<const char*>(&i32), 
              sizeof(int32_t));
          }
          break;
        case array::t_int64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],i64);
            m_out_stream.write( reinterpret_cast<const char*>(&i64), 
              sizeof(int64_t));
          }
          break;
        case array::t_uint8:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui8);
            m_out_stream.write( reinterpret_cast<const char*>(&ui8), 
              sizeof(uint8_t));
          }
          break;
        case array::t_uint16:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui16);
            m_out_stream.write( reinterpret_cast<const char*>(&ui16), 
              sizeof(uint16_t));
          }
          break;
        case array::t_uint32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui32);
            m_out_stream.write( reinterpret_cast<const char*>(&ui32), 
              sizeof(uint32_t));
          }
          break;
        case array::t_uint64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ui64);
            m_out_stream.write( reinterpret_cast<const char*>(&ui64), 
              sizeof(uint64_t));
          }
          break;
        case array::t_float32:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],f);
            m_out_stream.write( reinterpret_cast<const char*>(&f), 
              sizeof(float));
          }
          break;
        case array::t_float64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],d);
            m_out_stream.write( reinterpret_cast<const char*>(&d), 
              sizeof(double));
          }
          break;
        case array::t_float128:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],ld);
            m_out_stream.write( reinterpret_cast<const char*>(&ld), 
              sizeof(long double));
          }
          break;
        case array::t_complex64:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],cf);
            m_out_stream.write( reinterpret_cast<const char*>(&cf), 
              sizeof(std::complex<float>));
          }
          break;
        case array::t_complex128:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],cd);
            m_out_stream.write( reinterpret_cast<const char*>(&cd), 
              sizeof(std::complex<double>));
          }
          break;
        case array::t_complex256:
          for( size_t i=0; i<m_header.getNElements(); ++i) {
            static_complex_cast(multi_array[i],cld);
            m_out_stream.write( reinterpret_cast<const char*>(&cld), 
              sizeof(std::complex<long double>));
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
    void BinOutputFile::write(const blitz::Array<T,D>& bl) {
      // Initialize the header if required
      if(!m_header_init)
        initHeader(bl);

      // Check the shape compatibility
      bool shapeCompatibility = true;
      size_t i=0;
      const size_t* h_shape = m_header.getShape();
      while( i<array::N_MAX_DIMENSIONS_ARRAY && shapeCompatibility) {
        if( i<D)
          shapeCompatibility = (static_cast<size_t>(bl.extent(i))==h_shape[i]);
        else
          shapeCompatibility = (h_shape[i] == 0);
        ++i;
      }

      if(!shapeCompatibility)
      {
        error << "The dimensions of this array does not match the " <<
          "ones contained in the header file. The array cannot be saved." <<
          std::endl;
        throw Exception();
      }

      // Copy the data into the output stream
      const T* data;
      if( bl.isStorageContiguous() )
        data = bl.data();
      else
        data = bl.copy().data();

      if(m_header.needCast(bl))
        writeWithCast(data);
      else
        write(data);
    }


    template <typename T, int d>
    void BinOutputFile::initHeader(const blitz::Array<T,d>& bl)
    {
      // Check that data have not already been written
      if( m_n_arrays_written > 0 ) { 
        error << "Cannot init the header of an output stream in which data" <<
          " have already been written." << std::endl;
        throw Exception();
      }   
    
      // Initialize header
      initTypeHeader(bl);
      size_t shape[array::N_MAX_DIMENSIONS_ARRAY];
      for(size_t i=0; i<array::N_MAX_DIMENSIONS_ARRAY; ++i) {
        shape[i] = ( i<d ? bl.extent(i) : 0);
      }
      m_header.setShape(shape);
      m_header.write(m_out_stream);
      m_header_init = true;
    }

    template <typename T, int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<T,d>& bl)
    {
      error << "Unsupported blitz array type " << std::endl;
      throw TypeError();
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<bool,d>& bl)
    {
      m_header.setArrayType(array::t_bool);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<int8_t,d>& bl)
    {
      m_header.setArrayType(array::t_int8);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<int16_t,d>& bl)
    {
      m_header.setArrayType(array::t_int16);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<int32_t,d>& bl)
    {
      m_header.setArrayType(array::t_int32);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<int64_t,d>& bl)
    {
      m_header.setArrayType(array::t_int64);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<uint8_t,d>& bl)
    {
      m_header.setArrayType(array::t_uint8);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<uint16_t,d>& bl)
    {
      m_header.setArrayType(array::t_uint16);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<uint32_t,d>& bl)
    {
      m_header.setArrayType(array::t_uint32);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<uint64_t,d>& bl)
    {
      m_header.setArrayType(array::t_uint64);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<float,d>& bl)
    {
      m_header.setArrayType(array::t_float32);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<double,d>& bl)
    {
      m_header.setArrayType(array::t_float64);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(const blitz::Array<long double,d>& bl)
    {
      m_header.setArrayType(array::t_float128);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(
      const blitz::Array<std::complex<float>,d>& bl)
    {
      m_header.setArrayType(array::t_complex64);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(
      const blitz::Array<std::complex<double>,d>& bl)
    {
      m_header.setArrayType(array::t_complex128);
    }

    template <int d>
    void BinOutputFile::initTypeHeader(
      const blitz::Array<std::complex<long double>,d>& bl)
    {
      m_header.setArrayType(array::t_complex256);
    }

  }
}

#endif /* TORCH5SPRO_CORE_BIN_OUTPUT_FILE_H */

