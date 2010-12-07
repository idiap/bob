/**
 * @file src/core/src/Dataset2.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the Dataset class.
 */

#include "core/Dataset2.h"
#include "core/Exception.h"

namespace Torch {
  namespace core {

    Array::Array(const boost::shared_ptr<Arrayset> parent): 
      m_parent_arrayset(parent), m_id(0), m_is_loaded(false), m_filename(""),
      m_loader(l_unknown), m_storage(0) { }

    Array::~Array() {
      delete [] m_storage;
    } 


    Arrayset::Arrayset(): 
      m_id(0), m_n_dim(0), m_element_type(t_unknown), m_role(""), 
      m_is_loaded(false), m_filename(""), m_loader(l_unknown) 
    {
      m_shape[0]=m_shape[1]=m_shape[2]=m_shape[3]=0; 
    }

    Arrayset::~Arrayset() { } 

    void Arrayset::add_array( boost::shared_ptr<Array> array) {
      m_array.insert( std::pair<size_t,boost::shared_ptr<Array> >(
        array->getId(), boost::shared_ptr<Array>(array)) );
    }

    template<typename T, int D> void 
      Arrayset::at(size_t id, blitz::Array<T,D>& output) {
      // TODO: to implement
    }


    Dataset::Dataset() { }

    Dataset::~Dataset() { }

    void Dataset::add_arrayset( boost::shared_ptr<Arrayset> arrayset) {
      m_arrayset.insert( std::pair<size_t,boost::shared_ptr<Arrayset> >(
        arrayset->getId(), boost::shared_ptr<Arrayset>(arrayset)) );
    }

    const boost::shared_ptr<Arrayset> Dataset::at( const size_t id ) const {
      return (m_arrayset.find(id))->second;
    }


  }
}

