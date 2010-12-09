/**
 * @file src/core/src/Dataset2.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the Dataset class.
 */

#include "core/Dataset2.h"

namespace Torch {
  namespace core {

    Array::Array(const Arrayset& parent): 
      m_parent_arrayset(parent), m_id(0), m_is_loaded(false), m_filename(""),
      m_loader(l_unknown), m_storage(0) { }

    Array::~Array() {
      std::cout << "Array destructor (id: " << getId() << ")" << std::endl;
      switch(m_parent_arrayset.getArrayType()/* m_element_type*/) {
        case t_bool:
          delete [] static_cast<bool*>(m_storage); break;
        case t_int8:
          delete [] static_cast<int8_t*>(m_storage); break;
        case t_int16:
          delete [] static_cast<int16_t*>(m_storage); break;
        case t_int32:
          delete [] static_cast<int32_t*>(m_storage); break;
        case t_int64:
          delete [] static_cast<int64_t*>(m_storage); break;
        case t_uint8:
          delete [] static_cast<uint8_t*>(m_storage); break;
        case t_uint16:
          delete [] static_cast<uint16_t*>(m_storage); break;
        case t_uint32:
          delete [] static_cast<uint32_t*>(m_storage); break;
        case t_uint64:
          delete [] static_cast<uint64_t*>(m_storage); break;
        case t_float32:
          delete [] static_cast<float*>(m_storage); break;
        case t_float64:
          delete [] static_cast<double*>(m_storage); break;
        case t_float128:
          delete [] static_cast<long double*>(m_storage); break;
        case t_complex64:
          delete [] static_cast<std::complex<float>* >(m_storage); break;
        case t_complex128:
          delete [] static_cast<std::complex<double>* >(m_storage); break;
        case t_complex256:
          delete [] static_cast<std::complex<long double>* >(m_storage); 
          break;
        default:
          break;
      }
    } 


    Arrayset::Arrayset(): 
      m_id(0), m_n_dim(0), m_n_elem(0), m_element_type(t_unknown), m_role(""),
      m_is_loaded(false), m_filename(""), m_loader(l_unknown) 
    {
      m_shape[0]=m_shape[1]=m_shape[2]=m_shape[3]=0; 
    }

    Arrayset::~Arrayset() {
      std::cout << "Arrayset destructor (id: " << getId() << ")" << std::endl;
      for(iterator it=begin(); it!=end(); ++it)
        it->second.reset();
    }

    void Arrayset::addArray( boost::shared_ptr<Array> array) {
      m_array.insert( std::pair<size_t,boost::shared_ptr<Array> >(
        array->getId(), array) );
    }


    Dataset::Dataset() { }

    Dataset::~Dataset() {
      std::cout << "Dataset destructor" << std::endl;
      for(iterator it=begin(); it!=end(); ++it)
        it->second.reset();
    }

    void Dataset::addArrayset( boost::shared_ptr<Arrayset> arrayset) {
      m_arrayset.insert( std::pair<size_t,boost::shared_ptr<Arrayset> >(
        arrayset->getId(), arrayset) );
    }

    const boost::shared_ptr<Arrayset> Dataset::at( const size_t id ) const {
      return (m_arrayset.find(id))->second;
    }


  }
}

