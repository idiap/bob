/**
 * @file src/cxx/core/src/BinFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store and load multiarrays into/from files.
 */


#include "core/BinFile.h"

namespace Torch {
  namespace core {

    BinFile::BinFile(const std::string& filename, openmode flag):
      m_header_init(false),
      m_current_array(0),
      m_n_arrays_written(0),
      m_openmode(flag)
    {
      if((flag & BinFile::out) && (flag & BinFile::in)) {
        m_stream.open(filename.c_str(), std::ios::in | std::ios::out | 
          std::ios::binary);
        m_header.read(m_stream);
        m_header_init = true;
        m_n_arrays_written = m_header.m_n_samples;

        if( flag & BinFile::append) {
          m_stream.seekp(0, std::ios::end);
          m_current_array = m_header.m_n_samples;
        }
      }
      else if(flag & BinFile::out) {
        m_stream.open(filename.c_str(), std::ios::out | std::ios::binary);

        if( flag & BinFile::append) {
          m_header.read(m_stream);
          m_header_init = true;
          m_n_arrays_written = m_header.m_n_samples;
          m_stream.seekp(0, std::ios::end);
          m_current_array = m_header.m_n_samples;

        }
      }
      else if(flag & BinFile::in) {
        m_stream.open(filename.c_str(), std::ios::in | std::ios::binary);
        m_header.read(m_stream);
        m_header_init = true;
        m_n_arrays_written = m_header.m_n_samples;
        
        if( flag & BinFile::append) {
          error << "Cannot append data in read only mode." << std::endl;
          throw Exception();
        }
      }
      else
      {
        error << "Invalid combination of flags." << std::endl;
        throw Exception();
      }
    }

    BinFile::~BinFile() {
      close();
    }


    void BinFile::initHeader(const array::ElementType type, 
        const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]) 
    {
      // Check that data have not already been written
      if( m_n_arrays_written > 0 ) {
        error << "Cannot init the header of an output stream in which data" <<
          " have already been written." << std::endl;
        throw Exception();
      }
      
      // Initialize header
      m_header.m_elem_type = type;
      m_header.typeUpdated();
      m_header.setShape(shape);
      m_header.write(m_stream);
      m_header_init = true;
    }

    void BinFile::close() {
      // Rewrite the header and update the number of samples
      m_header.m_n_samples = m_n_arrays_written;
      if(m_openmode & BinFile::out)
        m_header.write(m_stream);

      m_stream.close();
    }


    void BinFile::write(const Arrayset& arrayset) {
      // Initialize the header if required
      if(!m_header_init)
        initHeader( arrayset.getElementType(), arrayset.getShape() );

      if(!arrayset.getIsLoaded()) {
        error << "The arrayset is not loaded." << std::endl;
        throw Exception();
      }
      
      for(Arrayset::const_iterator it=arrayset.begin(); it!=arrayset.end(); 
        ++it)
      {
        write(*(it->second));
      }
    }

    void BinFile::write(const Array& array) {
      // Initialize the header if required
      if(!m_header_init)
        initHeader( array.getParentArrayset().getElementType(), 
          array.getParentArrayset().getShape() );

      bool shapeCompatibility = true;
      size_t i=0;
      const size_t* p_shape = array.getParentArrayset().getShape();
      const size_t* h_shape = m_header.m_shape;
      while( i<array::N_MAX_DIMENSIONS_ARRAY && shapeCompatibility) {
        shapeCompatibility = ( p_shape[i] == h_shape[i]);
        ++i;
      }
        
      if(!shapeCompatibility)
      {
        error << "The dimensions of this array does not match the " <<
          "contained in the header file. The array cannot be saved." <<
          std::endl;
        throw Exception();
      }

      if(array.getParentArrayset().getElementType() == m_header.m_elem_type)
        write(array.getStorage()); 
      else // cast is required
      {
        // copy the data into the output stream
        switch(array.getParentArrayset().getElementType())
        {
          case array::t_bool:
            writeWithCast( reinterpret_cast<const bool*>(array.getStorage()) );
            break;
          case array::t_int8:
            writeWithCast( 
              reinterpret_cast<const int8_t*>(array.getStorage()) );
            break;
          case array::t_int16:
            writeWithCast( 
              reinterpret_cast<const int16_t*>(array.getStorage()) );
            break;
          case array::t_int32:
            writeWithCast( 
              reinterpret_cast<const int32_t*>(array.getStorage()) );
            break;
          case array::t_int64:
            writeWithCast( 
              reinterpret_cast<const int64_t*>(array.getStorage()) );
            break;
          case array::t_uint8:
            writeWithCast( 
              reinterpret_cast<const uint8_t*>(array.getStorage()) );
            break;
          case array::t_uint16:
            writeWithCast( 
              reinterpret_cast<const uint16_t*>(array.getStorage()) );
            break;
          case array::t_uint32:
            writeWithCast( 
              reinterpret_cast<const uint32_t*>(array.getStorage()) );
            break;
          case array::t_uint64:
            writeWithCast( 
              reinterpret_cast<const uint64_t*>(array.getStorage()) );
            break;
          case array::t_float32:
            writeWithCast( 
              reinterpret_cast<const float*>(array.getStorage()) );
            break;
          case array::t_float64:
            writeWithCast( 
              reinterpret_cast<const double*>(array.getStorage()) );
            break;
          case array::t_complex64:
            writeWithCast( reinterpret_cast<const std::complex<float>* >(
              array.getStorage()) );
            break;
          case array::t_complex128:
            writeWithCast( reinterpret_cast<const std::complex<double>*>(
              array.getStorage()) );
            break;
          default:
            break;
        }
      }
    }

    BinFile& BinFile::write(const void* multi_array) {
      // copy the data into the output stream
      switch(m_header.m_elem_type)
      {
        case array::t_bool:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(bool));
          break;
        case array::t_int8:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(int8_t));
          break;
        case array::t_int16:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(int16_t));
          break;
        case array::t_int32:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(int32_t));
          break;
        case array::t_int64:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(int64_t));
          break;
        case array::t_uint8:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(uint8_t));
          break;
        case array::t_uint16:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(uint16_t));
          break;
        case array::t_uint32:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(uint32_t));
          break;
        case array::t_uint64:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(uint64_t));
          break;
        case array::t_float32:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(float));
          break;
        case array::t_float64:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(double));
          break;
        case array::t_complex64:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(std::complex<float>));
          break;
        case array::t_complex128:
          m_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.m_n_elements*sizeof(std::complex<double>));
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

    void BinFile::read( Arrayset& arrayset) {
      // Create/allocate arrays inside the arrayset
      size_t n_arrays = m_header.m_n_samples - m_current_array;
      for( size_t i=0; i<n_arrays; ++i) {
        // Create a new array 
        boost::shared_ptr<Array> ar(new Array(arrayset));

        // Update some array members (m_id and m_is_loaded)
        ar->setId(i+1);
        ar->setIsLoaded(true);

        // Update the array with the content from the binary file
        read(*ar);
        
        // Add the array to the arrayset
        arrayset.append(ar);
      }
    }

    void BinFile::read( Array& array) {
      // Check that the last array was not reached in the binary file
      endOfFile();

      // Check shape compatibility between the binary file and the array
      bool shapeCompatibility = true;
      size_t i=0;
      const size_t* p_shape = array.getParentArrayset().getShape();
      const size_t* h_shape = m_header.m_shape;
      while( i<array::N_MAX_DIMENSIONS_ARRAY && shapeCompatibility) {
        shapeCompatibility = ( p_shape[i] == h_shape[i]);
        ++i;
      }
        
      if(!shapeCompatibility)
      {
        error << "The dimensions of this array does not match the " <<
          "ones contained in the header file (of the binary). " << 
          "The array cannot be loaded." << std::endl;
        throw Exception();
      }

      // Allocate memory for storing the array,
      // and copy the content from the binary file
      void* storage;
      switch(array.getParentArrayset().getElementType())
      {
        case array::t_bool:
          storage = new bool[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<bool*>(storage) );
          else
            readWithCast( reinterpret_cast<bool*>(storage) );
          break;
        case array::t_int8:
          storage = new int8_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<int8_t*>(storage) );
          else
            readWithCast( reinterpret_cast<int8_t*>(storage) );
          break;
        case array::t_int16:
          storage = new int16_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<int16_t*>(storage) );
          else
            readWithCast( reinterpret_cast<int16_t*>(storage) );
          break;
        case array::t_int32:
          storage = new int32_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<int32_t*>(storage) );
          else
            readWithCast( reinterpret_cast<int32_t*>(storage) );
          break;
        case array::t_int64:
          storage = new int64_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<int64_t*>(storage) );
          else
            readWithCast( reinterpret_cast<int64_t*>(storage) );
          break;
        case array::t_uint8:
          storage = new uint8_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<uint8_t*>(storage) );
          else
            readWithCast( reinterpret_cast<uint8_t*>(storage) );
          break;
        case array::t_uint16:
          storage = new uint16_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<uint16_t*>(storage) );
          else
            readWithCast( reinterpret_cast<uint16_t*>(storage) );
          break;
        case array::t_uint32:
          storage = new uint32_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<uint32_t*>(storage) );
          else
            readWithCast( reinterpret_cast<uint32_t*>(storage) );
          break;
        case array::t_uint64:
          storage = new uint64_t[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<uint64_t*>(storage) );
          else
            readWithCast( reinterpret_cast<uint64_t*>(storage) );
          break;
        case array::t_float32:
          storage = new float[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<float*>(storage) );
          else
            readWithCast( reinterpret_cast<float*>(storage) );
          break;
        case array::t_float64:
          storage = new double[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<double*>(storage) );
          else
            readWithCast( reinterpret_cast<double*>(storage) );
          break;
        case array::t_complex64:
          storage = new 
            std::complex<float>[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<std::complex<float>* >(storage) );
          else
            readWithCast( 
              reinterpret_cast<std::complex<double>* >(storage) );
          break;
        case array::t_complex128:
          storage = new 
            std::complex<double>[array.getParentArrayset().getNElem()];
          if(array.getParentArrayset().getElementType()==m_header.m_elem_type)
            read( reinterpret_cast<std::complex<double>* >(storage) );
          else
            readWithCast( 
              reinterpret_cast<std::complex<double>* >(storage) );
          break;
        default:
          break;
      }
      array.setStorage( storage);
      
      // Update the m_is_loaded member of the array
      array.setIsLoaded(true);
    }

    BinFile& BinFile::read(void* multi_array) {
      // copy the multiarray from the input stream to the C-style array
      switch(m_header.m_elem_type)
      {
        case array::t_bool:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(bool));
          break;
        case array::t_int8:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(int8_t));
          break;
        case array::t_int16:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(int16_t));
          break;
        case array::t_int32:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(int32_t));
          break;
        case array::t_int64:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(int64_t));
          break;
        case array::t_uint8:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(uint8_t));
          break;
        case array::t_uint16:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(uint16_t));
          break;
        case array::t_uint32:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(uint32_t));
          break;
        case array::t_uint64:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(uint64_t));
          break;
        case array::t_float32:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(float));
          break;
        case array::t_float64:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(double));
          break;
        case array::t_complex64:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(std::complex<float>));
          break;
        case array::t_complex128:
          m_stream.read( reinterpret_cast<char*>(multi_array),
            m_header.m_n_elements*sizeof(std::complex<double>));
          break;
        default:
          break;
      }

      // Update current array
      ++m_current_array;

      return *this;
    }

/**
 * @brief Specialization of the initTypeHeader(), which sets the
 * element type in the header
 */
#define INIT_HEADER_DEF(T,name,D) template<> \
    void BinFile::initTypeHeader(const blitz::Array<T,D>& bl) \
    { \
      m_header.m_elem_type = name; \
      m_header.typeUpdated(); \
    } \
 
    INIT_HEADER_DEF(bool,array::t_bool,1)
    INIT_HEADER_DEF(bool,array::t_bool,2)
    INIT_HEADER_DEF(bool,array::t_bool,3)
    INIT_HEADER_DEF(bool,array::t_bool,4)
    INIT_HEADER_DEF(int8_t,array::t_int8,1)
    INIT_HEADER_DEF(int8_t,array::t_int8,2)
    INIT_HEADER_DEF(int8_t,array::t_int8,3)
    INIT_HEADER_DEF(int8_t,array::t_int8,4)
    INIT_HEADER_DEF(int16_t,array::t_int16,1)
    INIT_HEADER_DEF(int16_t,array::t_int16,2)
    INIT_HEADER_DEF(int16_t,array::t_int16,3)
    INIT_HEADER_DEF(int16_t,array::t_int16,4)
    INIT_HEADER_DEF(int32_t,array::t_int32,1)
    INIT_HEADER_DEF(int32_t,array::t_int32,2)
    INIT_HEADER_DEF(int32_t,array::t_int32,3)
    INIT_HEADER_DEF(int32_t,array::t_int32,4)
    INIT_HEADER_DEF(int64_t,array::t_int64,1)
    INIT_HEADER_DEF(int64_t,array::t_int64,2)
    INIT_HEADER_DEF(int64_t,array::t_int64,3)
    INIT_HEADER_DEF(int64_t,array::t_int64,4)
    INIT_HEADER_DEF(uint8_t,array::t_uint8,1)
    INIT_HEADER_DEF(uint8_t,array::t_uint8,2)
    INIT_HEADER_DEF(uint8_t,array::t_uint8,3)
    INIT_HEADER_DEF(uint8_t,array::t_uint8,4)
    INIT_HEADER_DEF(uint16_t,array::t_uint16,1)
    INIT_HEADER_DEF(uint16_t,array::t_uint16,2)
    INIT_HEADER_DEF(uint16_t,array::t_uint16,3)
    INIT_HEADER_DEF(uint16_t,array::t_uint16,4)
    INIT_HEADER_DEF(uint32_t,array::t_uint32,1)
    INIT_HEADER_DEF(uint32_t,array::t_uint32,2)
    INIT_HEADER_DEF(uint32_t,array::t_uint32,3)
    INIT_HEADER_DEF(uint32_t,array::t_uint32,4)
    INIT_HEADER_DEF(uint64_t,array::t_uint64,1)
    INIT_HEADER_DEF(uint64_t,array::t_uint64,2)
    INIT_HEADER_DEF(uint64_t,array::t_uint64,3)
    INIT_HEADER_DEF(uint64_t,array::t_uint64,4)
    INIT_HEADER_DEF(float,array::t_float32,1)
    INIT_HEADER_DEF(float,array::t_float32,2)
    INIT_HEADER_DEF(float,array::t_float32,3)
    INIT_HEADER_DEF(float,array::t_float32,4)
    INIT_HEADER_DEF(double,array::t_float64,1)
    INIT_HEADER_DEF(double,array::t_float64,2)
    INIT_HEADER_DEF(double,array::t_float64,3)
    INIT_HEADER_DEF(double,array::t_float64,4)
    INIT_HEADER_DEF(std::complex<float>,array::t_complex64,1)
    INIT_HEADER_DEF(std::complex<float>,array::t_complex64,2)
    INIT_HEADER_DEF(std::complex<float>,array::t_complex64,3)
    INIT_HEADER_DEF(std::complex<float>,array::t_complex64,4)
    INIT_HEADER_DEF(std::complex<double>,array::t_complex128,1)
    INIT_HEADER_DEF(std::complex<double>,array::t_complex128,2)
    INIT_HEADER_DEF(std::complex<double>,array::t_complex128,3)
    INIT_HEADER_DEF(std::complex<double>,array::t_complex128,4)


/**
 * @brief Specializations of the writeBlitz() function, which write
 * the data from a blitz array into a binary file.
 * @warning It assumes that the shape and the type of the blitz array
 * match the ones of the binary file.
 */
#define WRITE_BLITZ(T) \
  template<> void BinFile::writeBlitz(const blitz::Array<T,1>& bl) { \
    T val; \
    for(int i=0; i<bl.extent(0); ++i) { \
      val = bl(i); \
      m_stream.write( reinterpret_cast<const char*>(&val), sizeof(T)); \
    } \
    ++m_current_array; \
    if(m_current_array>m_n_arrays_written) \
      ++m_n_arrays_written; \
  } \
\
  template<> void BinFile::writeBlitz(const blitz::Array<T,2>& bl) { \
    T val; \
    for(int i=0; i<bl.extent(0); ++i) { \
      for(int j=0; j<bl.extent(1); ++j) { \
        val = bl(i,j); \
        m_stream.write( reinterpret_cast<const char*>(&val), sizeof(T)); \
      } \
    } \
    ++m_current_array; \
    if(m_current_array>m_n_arrays_written) \
      ++m_n_arrays_written; \
  } \
\
  template<> void BinFile::writeBlitz(const blitz::Array<T,3>& bl) { \
    T val; \
    for(int i=0; i<bl.extent(0); ++i) { \
      for(int j=0; j<bl.extent(1); ++j) { \
        for(int k=0; k<bl.extent(2); ++k) { \
          val = bl(i,j,k); \
          m_stream.write( reinterpret_cast<const char*>(&val), sizeof(T)); \
        } \
      } \
    } \
    ++m_current_array; \
    if(m_current_array>m_n_arrays_written) \
      ++m_n_arrays_written; \
  } \
\
  template<> void BinFile::writeBlitz(const blitz::Array<T,4>& bl) { \
    T val; \
    for(int i=0; i<bl.extent(0); ++i) { \
      for(int j=0; j<bl.extent(1); ++j) { \
        for(int k=0; k<bl.extent(2); ++k) { \
          for(int l=0; l<bl.extent(3); ++l) { \
            val = bl(i,j,k,l); \
            m_stream.write( reinterpret_cast<const char*>(&val), sizeof(T)); \
          } \
        } \
      } \
    } \
    ++m_current_array; \
    if(m_current_array>m_n_arrays_written) \
      ++m_n_arrays_written; \
  } \
\

    WRITE_BLITZ(bool)
    WRITE_BLITZ(int8_t)
    WRITE_BLITZ(int16_t)
    WRITE_BLITZ(int32_t)
    WRITE_BLITZ(int64_t)
    WRITE_BLITZ(uint8_t)
    WRITE_BLITZ(uint16_t)
    WRITE_BLITZ(uint32_t)
    WRITE_BLITZ(uint64_t)
    WRITE_BLITZ(float)
    WRITE_BLITZ(double)
    WRITE_BLITZ(std::complex<float>)
    WRITE_BLITZ(std::complex<double>)

  }
}

