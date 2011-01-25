/**
 * @file src/cxx/core/src/BinOutputFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store multiarrays into files.
 */


#include "core/BinOutputFile.h"

namespace Torch {
  namespace core {

    BinOutputFile::BinOutputFile(const std::string& filename, bool append):
      m_header_init(append), 
      m_out_stream( filename.c_str(), std::ios::out | std::ios::binary ),
      m_n_arrays_written(0)
    {
      if(append) {
        m_header.read(m_out_stream);
        m_header_init = true;
        m_n_arrays_written = m_header.getNSamples();
        m_out_stream.seekp(0, std::ios::end);
      }
    }

    BinOutputFile::~BinOutputFile() {
      close();
    }


    void BinOutputFile::initHeader(const array::ElementType type, 
        const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]) 
    {
      // Check that data have not already been written
      if( m_n_arrays_written > 0 ) {
        error << "Cannot init the header of an output stream in which data" <<
          " have already been written." << std::endl;
        throw Exception();
      }
      
      // Initialize header
      m_header.setElementType( type);
      m_header.setShape( shape);
      m_header.write(m_out_stream);
      m_header_init = true;
    }

    void BinOutputFile::close() {
      // Rewrite the header and update the number of samples
      m_header.setNSamples(m_n_arrays_written);
      m_header.write(m_out_stream);

      m_out_stream.close();
    }


    void BinOutputFile::write(const Arrayset& arrayset) {
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

    void BinOutputFile::write(const Array& array) {
      // Initialize the header if required
      if(!m_header_init)
        initHeader( array.getParentArrayset().getElementType(), 
          array.getParentArrayset().getShape() );

      bool shapeCompatibility = true;
      size_t i=0;
      const size_t* p_shape = array.getParentArrayset().getShape();
      const size_t* h_shape = m_header.getShape();
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

      if(array.getParentArrayset().getElementType() == m_header.getElementType())
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
          case array::t_float128:
            writeWithCast( 
              reinterpret_cast<const long double*>(array.getStorage()) );
            break;
          case array::t_complex64:
            writeWithCast( reinterpret_cast<const std::complex<float>* >(
              array.getStorage()) );
            break;
          case array::t_complex128:
            writeWithCast( reinterpret_cast<const std::complex<double>*>(
              array.getStorage()) );
            break;
          case array::t_complex256:
            writeWithCast( reinterpret_cast<const std::complex<long double>* >(
              array.getStorage()) );
            break;
          default:
            break;
        }
      }
    }

    BinOutputFile& BinOutputFile::write(const void* multi_array) {
      // copy the data into the output stream
      switch(m_header.getElementType())
      {
        case array::t_bool:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(bool));
          break;
        case array::t_int8:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(int8_t));
          break;
        case array::t_int16:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(int16_t));
          break;
        case array::t_int32:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(int32_t));
          break;
        case array::t_int64:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(int64_t));
          break;
        case array::t_uint8:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(uint8_t));
          break;
        case array::t_uint16:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(uint16_t));
          break;
        case array::t_uint32:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(uint32_t));
          break;
        case array::t_uint64:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(uint64_t));
          break;
        case array::t_float32:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(float));
          break;
        case array::t_float64:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(double));
          break;
        case array::t_float128:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(long double));
          break;
        case array::t_complex64:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(std::complex<float>));
          break;
        case array::t_complex128:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(std::complex<double>));
          break;
        case array::t_complex256:
          m_out_stream.write( reinterpret_cast<const char*>(multi_array), 
            m_header.getNElements()*sizeof(std::complex<long double>));
          break;
        default:
          break;
      }

      // increment m_n_arrays_written
      ++m_n_arrays_written;

      return *this;
    }


  }
}

