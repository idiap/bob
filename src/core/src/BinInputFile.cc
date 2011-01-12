/**
 * @file src/core/src/BinInputFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to load multiarrays from files.
 */


#include "core/BinInputFile.h"

namespace Torch {
  namespace core {

    BinInputFile::BinInputFile(const std::string& filename): 
      m_current_array(0),
      m_in_stream( filename.c_str(), std::ios::in | std::ios::binary ) 
    {
      m_header.read(m_in_stream);
    }

    BinInputFile::~BinInputFile() {
      close();
    }

    void BinInputFile::close() {
      m_in_stream.close();
    } 

    void BinInputFile::load( Arrayset& arrayset) {
      //TODO: implementation:
      // There is a need to create/allocate arrays inside the arrayset
    }

    void BinInputFile::load( Array& array) {
      // Check shape compatibility between the binary file and the array
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
          "ones contained in the header file (of the binary). " << 
          "The array cannot be loaded." << std::endl;
        throw Exception();
      }

      // Allocate memory for storing the array,
      // and copy the content from the binary file
      void* storage;
      switch(array.getParentArrayset().getArrayType())
      {
        case array::t_bool:
          storage = new bool[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<bool*>(storage) );
          break;
        case array::t_int8:
          storage = new int8_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<int8_t*>(storage) );
          break;
        case array::t_int16:
          storage = new int16_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<int16_t*>(storage) );
          break;
        case array::t_int32:
          storage = new int32_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<int32_t*>(storage) );
          break;
        case array::t_int64:
          storage = new int64_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<int64_t*>(storage) );
          break;
        case array::t_uint8:
          storage = new uint8_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<uint8_t*>(storage) );
          break;
        case array::t_uint16:
          storage = new uint16_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<uint16_t*>(storage) );
          break;
        case array::t_uint32:
          storage = new uint32_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<uint32_t*>(storage) );
          break;
        case array::t_uint64:
          storage = new uint64_t[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<uint64_t*>(storage) );
          break;
        case array::t_float32:
          storage = new float[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<float*>(storage) );
          break;
        case array::t_float64:
          storage = new double[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<double*>(storage) );
          break;
        case array::t_float128:
          storage = new long double[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<long double*>(storage) );
          break;
        case array::t_complex64:
          storage = new 
            std::complex<float>[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<std::complex<float>* >(storage) );
          break;
        case array::t_complex128:
          storage = new 
            std::complex<double>[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<std::complex<double>* >(storage) );
          break;
        case array::t_complex256:
          storage = new 
            std::complex<long double>[array.getParentArrayset().getNElem()];
          operator>>( reinterpret_cast<std::complex<long double>* >(storage) );
          break;
        default:
          break;
      }
      array.setStorage( storage);
      
      // Update the m_is_loaded member of the array
      array.setIsLoaded(true);
    }

  }
}

