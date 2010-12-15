/**
 * @file src/core/src/BinOutputFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to store multiarrays into files.
 */


#include "core/BinOutputFile.h"

namespace Torch {
  namespace core {

    BinOutputFile::BinOutputFile(const std::string& filename, bool append):
      m_write_init(append), 
      m_out_stream( filename.c_str(), std::ios::out | std::ios::binary ),
      m_n_arrays_written(0)
    {
      if(append) {
        m_header.read(m_out_stream);
        m_n_arrays_written = m_header.getNSamples();
        m_out_stream.seekp(0, std::ios::end);
      }
    }


    BinOutputFile::~BinOutputFile() {
      close();
    }


    void BinOutputFile::initHeader(const array::ArrayType type, 
        const size_t shape[array::N_MAX_DIMENSIONS_ARRAY]) 
    {
      m_header.setArrayType( type);
      m_header.setShape( shape);
      m_header.write(m_out_stream);
      m_write_init = true;
    }

    void BinOutputFile::close()
    {
      m_header.setNSamples(m_n_arrays_written);
      m_out_stream.close();
    }


    void BinOutputFile::save(const Arrayset& arrayset) {
      // Check that the header has been initialized
      checkWriteInit();

      if(!arrayset.getIsLoaded()) {
        error << "The arrayset is not loaded." << std::endl;
        throw Exception();
      }

      initHeader( arrayset.getArrayType(), arrayset.getShape() );
      
      for(Arrayset::const_iterator it=arrayset.begin(); it!=arrayset.end(); 
        ++it)
      {
        save(*(it->second));
      }
    }

    void BinOutputFile::save(const Array& array) {
      // Check that the header has been initialized
      checkWriteInit();
      
      bool shapeCompatibility = true;
      size_t i=0;
      while( i<array::N_MAX_DIMENSIONS_ARRAY && shapeCompatibility) {
        shapeCompatibility = ( array.getParentArrayset().getShape()[i] ==
          getHeader().getShape()[i]);
        ++i;
      }
        
      if( shapeCompatibility && (array.getParentArrayset().getArrayType() != 
        getHeader().getArrayType() ) )
      {
        operator<<(array.getStorage()); 
      }
    }

    BinOutputFile& BinOutputFile::operator<<(const void* multi_array) {
      // Check that the header has been initialized
      checkWriteInit();

      // copy the data into the output stream
      switch(m_header.getArrayType())
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


    void BinOutputFile::checkWriteInit() {
      if(!m_write_init) {
        error << "The header have not yet been initialized with the type " <<
          "and dimensions" << std::endl;
        throw Exception();
      }
    }

  }
}

