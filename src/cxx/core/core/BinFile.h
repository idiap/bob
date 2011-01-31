/**
 * @file src/cxx/core/core/BinFile.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to load and store multiarrays from binary files.
 */

#ifndef TORCH5SPRO_CORE_BIN_FILE_H
#define TORCH5SPRO_CORE_BIN_FILE_H 1

#include "core/BinFileHeader.h"
#include "core/Dataset2.h"
#include "core/StaticComplexCast.h"

namespace Torch {
  namespace core {

    /**
     * @brief Defines the flags that might be used when loading/storing
     * a file containing blitz arrays.
     */
    enum _BinFileFlag {
      _append  = 1L << 0,
      _in      = 1L << 3,
      _out     = 1L << 4
    };

    inline _BinFileFlag operator&(_BinFileFlag a, _BinFileFlag b) { 
      return _BinFileFlag(static_cast<int>(a) & static_cast<int>(b)); 
    }

    inline _BinFileFlag operator|(_BinFileFlag a, _BinFileFlag b) { 
      return _BinFileFlag(static_cast<int>(a) | static_cast<int>(b)); 
    }

    inline _BinFileFlag operator^(_BinFileFlag a, _BinFileFlag b) { 
      return _BinFileFlag(static_cast<int>(a) ^ static_cast<int>(b)); 
    }

    inline _BinFileFlag& operator|=(_BinFileFlag& a, _BinFileFlag b) { 
      return a = a | b; 
    }

    inline _BinFileFlag& operator&=(_BinFileFlag& a, _BinFileFlag b) { 
      return a = a & b; 
    }

    inline _BinFileFlag& operator^=(_BinFileFlag& a, _BinFileFlag b) { 
      return a = a ^ b; 
    }

    inline _BinFileFlag operator~(_BinFileFlag a) { 
      return _BinFileFlag(~static_cast<int>(a)); 
    }


    /**
     *  @brief This class can be used for loading and storing multiarrays
     *  from/to binary files
     */
    class BinFile
    {
      public:
        /**
         * @brief Define the bitmask type for providing information
         * about the type of the stream.
         */
        typedef _BinFileFlag openmode;
        static const openmode append  = _append;
        static const openmode in      = _in;
        static const openmode out     = _out; 

        /**
         * @brief Constructor
         */
        BinFile(const std::string& filename, openmode f);

        /**
         * @brief Destructor
         */
        ~BinFile();

        /**
         * Close the BinFile
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
         * @warning An exception is thrown if nothing was written so far
         */
        array::ElementType getElementType() const { 
          headerInitialized(); 
          return m_header.m_elem_type; 
        }
        /**
         * @brief Get the number of dimensions
         * @warning An exception is thrown if nothing was written so far
         */
        size_t getNDimensions() const {  
          headerInitialized(); 
          return m_header.m_n_dimensions; 
        }
        /**
         * @brief Get the shape of each array
         * @warning An exception is thrown if nothing was written so far
         */
        const size_t* getShape() const { 
          headerInitialized(); 
          return m_header.m_shape; 
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
          return m_header.m_n_elements; 
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
        BinFile& write(const void* multi_array);

        /** 
         * @brief Put a C-style multiarray of a given type into the output
         * stream/file by casting it to the correct type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions.
         */
        template <typename T> 
        BinFile& writeWithCast(const T* multi_array);

        /**
         * @brief Put a void C-style multiarray into the output stream/file
         * @warning This is the responsability of the user to check
         * the correctness of the type and size of the memory block 
         * pointed by the void pointer
         */
        BinFile& read(void* multi_array);

        /**
         * @brief Get one C-style array from the input stream/file, and cast
         * it to the given type.
         * @warning The C-style array has to be allocated with the proper 
         * dimensions
         */
        template <typename T> BinFile& readWithCast(T* multiarray);

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
         * @brief Initialize the header of the (output) stream with the given
         * type and shape
         */
        void initHeader(const array::ElementType type, 
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
        void initTypeHeader(const blitz::Array<std::complex<float>,D>& bl);
        template<int D>
        void initTypeHeader(const blitz::Array<std::complex<double>,D>& bl);


       /**
         * Attributes
         */
        bool m_header_init;
        size_t m_current_array;
        size_t m_n_arrays_written;
        std::fstream m_stream;
        BinFileHeader m_header;
        openmode m_openmode;
    };


    template <typename T> 
    BinFile& BinFile::writeWithCast( const T* multi_array)
    {
      // copy the data into the output stream
      bool b;
      int8_t i8; int16_t i16; int32_t i32; int64_t i64;
      uint8_t ui8; uint16_t ui16; uint32_t ui32; uint64_t ui64;
      float f; double d; 
      std::complex<float> cf; std::complex<double> cd; 
      switch(m_header.m_elem_type)
      {
        case array::t_bool:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],b);
            m_stream.write( reinterpret_cast<const char*>(&b), 
              sizeof(bool));
          }
          break;
        case array::t_int8:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],i8);
            m_stream.write( reinterpret_cast<const char*>(&i8), 
              sizeof(int8_t));
          }
          break;
        case array::t_int16:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],i16);
            m_stream.write( reinterpret_cast<const char*>(&i16), 
              sizeof(int16_t));
          }
          break;
        case array::t_int32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],i32);
            m_stream.write( reinterpret_cast<const char*>(&i32), 
              sizeof(int32_t));
          }
          break;
        case array::t_int64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],i64);
            m_stream.write( reinterpret_cast<const char*>(&i64), 
              sizeof(int64_t));
          }
          break;
        case array::t_uint8:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],ui8);
            m_stream.write( reinterpret_cast<const char*>(&ui8), 
              sizeof(uint8_t));
          }
          break;
        case array::t_uint16:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],ui16);
            m_stream.write( reinterpret_cast<const char*>(&ui16), 
              sizeof(uint16_t));
          }
          break;
        case array::t_uint32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],ui32);
            m_stream.write( reinterpret_cast<const char*>(&ui32), 
              sizeof(uint32_t));
          }
          break;
        case array::t_uint64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],ui64);
            m_stream.write( reinterpret_cast<const char*>(&ui64), 
              sizeof(uint64_t));
          }
          break;
        case array::t_float32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],f);
            m_stream.write( reinterpret_cast<const char*>(&f), 
              sizeof(float));
          }
          break;
        case array::t_float64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],d);
            m_stream.write( reinterpret_cast<const char*>(&d), 
              sizeof(double));
          }
          break;
        case array::t_complex64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],cf);
            m_stream.write( reinterpret_cast<const char*>(&cf), 
              sizeof(std::complex<float>));
          }
          break;
        case array::t_complex128:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            static_complex_cast(multi_array[i],cd);
            m_stream.write( reinterpret_cast<const char*>(&cd), 
              sizeof(std::complex<double>));
          }
          break;
        default:
          break;
      }

      // increment m_n_arrays_written and m_current_array
      ++m_current_array;
      if(m_current_array>m_n_arrays_written)
        ++m_n_arrays_written;

      return *this;
    }


    template <typename T, int D> 
    void BinFile::write(const blitz::Array<T,D>& bl) {
      // Initialize the header if required
      if(!m_header_init)
        initHeader(bl);

      // Check the shape compatibility
      bool shapeCompatibility = true;
      size_t i=0;
      const size_t* h_shape = m_header.m_shape;
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
    void BinFile::read( blitz::Array<T,d>& bl) {
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
    void BinFile::read( size_t index, blitz::Array<T,d>& bl) {
      // Check that we are reaching an existing array
      if( index > m_header.m_n_samples ) {
        error << "Trying to reach a non-existing array." << std::endl;
        throw Exception();
      }

      // Set the stream pointer at the correct position
      m_stream.seekg( m_header.getArrayIndex(index) );
      m_current_array = index;

      // Put the content of the stream in the blitz array.
      read(bl);
    }

    template <typename T> 
    BinFile& BinFile::readWithCast(T* multiarray) {
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
            m_stream.read( reinterpret_cast<char*>(&b), sizeof(bool));
            static_complex_cast(b,multiarray[i]);
          }
          break;
        case array::t_int8:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&i8), sizeof(int8_t));
            static_complex_cast(i8,multiarray[i]);
          }
          break;
        case array::t_int16:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&i16), sizeof(int16_t));
            static_complex_cast(i16,multiarray[i]);
          }
          break;
        case array::t_int32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&i32), sizeof(int32_t));
            static_complex_cast(i32,multiarray[i]);
          }
          break;
        case array::t_int64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&i64), sizeof(int64_t));
            static_complex_cast(i64,multiarray[i]);
          }
          break;
        case array::t_uint8:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&ui8), sizeof(uint8_t));
            static_complex_cast(ui8,multiarray[i]);
          }
          break;
        case array::t_uint16:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&ui16), 
              sizeof(uint16_t));
            static_complex_cast(ui16,multiarray[i]);
          }
          break;
        case array::t_uint32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&ui32), 
              sizeof(uint32_t));
            static_complex_cast(ui32,multiarray[i]);
          }
          break;
        case array::t_uint64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&ui64), 
              sizeof(uint64_t));
            static_complex_cast(ui64,multiarray[i]);
          }
          break;
        case array::t_float32:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&f), sizeof(float));
            static_complex_cast(f,multiarray[i]);
          }
          break;
        case array::t_float64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&dou), sizeof(double));
            static_complex_cast(dou,multiarray[i]);
          }
          break;
        case array::t_complex64:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&cf), 
              sizeof(std::complex<float>));
            static_complex_cast(cf,multiarray[i]);
          }
          break;
        case array::t_complex128:
          for( size_t i=0; i<m_header.m_n_elements; ++i) {
            m_stream.read( reinterpret_cast<char*>(&cd), 
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


    template <typename T, int d>
    void BinFile::initHeader(const blitz::Array<T,d>& bl)
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
      m_header.write(m_stream);
      m_header_init = true;
    }

    template <typename T, int d>
    void BinFile::initTypeHeader(const blitz::Array<T,d>& bl)
    {
      error << "Unsupported blitz array type " << std::endl;
      throw TypeError();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<bool,d>& bl)
    {
      m_header.m_elem_type = array::t_bool;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<int8_t,d>& bl)
    {
      m_header.m_elem_type = array::t_int8;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<int16_t,d>& bl)
    {
      m_header.m_elem_type = array::t_int16;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<int32_t,d>& bl)
    {
      m_header.m_elem_type = array::t_int32;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<int64_t,d>& bl)
    {
      m_header.m_elem_type = array::t_int64;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<uint8_t,d>& bl)
    {
      m_header.m_elem_type = array::t_uint8;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<uint16_t,d>& bl)
    {
      m_header.m_elem_type = array::t_uint16;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<uint32_t,d>& bl)
    {
      m_header.m_elem_type = array::t_uint32;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<uint64_t,d>& bl)
    {
      m_header.m_elem_type = array::t_uint64;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<float,d>& bl)
    {
      m_header.m_elem_type = array::t_float32;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(const blitz::Array<double,d>& bl)
    {
      m_header.m_elem_type = array::t_float64;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(
      const blitz::Array<std::complex<float>,d>& bl)
    {
      m_header.m_elem_type = array::t_complex64;
      m_header.typeUpdated();
    }

    template <int d>
    void BinFile::initTypeHeader(
      const blitz::Array<std::complex<double>,d>& bl)
    {
      m_header.m_elem_type = array::t_complex128;
      m_header.typeUpdated();
    }

  }
}

#endif /* TORCH5SPRO_CORE_BIN_FILE_H */

