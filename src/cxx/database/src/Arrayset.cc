/**
 * @file src/cxx/database/src/Arrayset.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of an Array for a Dataset.
 */

#include "database/Arrayset.h"

namespace Torch {
  namespace database {

    Arrayset::Arrayset(boost::shared_ptr<const Dataset> parent, 
      const size_t id, const std::string& filename, const std::string& codec):
      m_parent_dataset(parent), m_id(id), m_n_dim(0), m_n_elem(0), 
      m_element_type(array::t_unknown), m_role(""), m_is_loaded(false), 
      m_filename(filename), m_codecname(codec)
    {
      m_shape[0]=m_shape[1]=m_shape[2]=m_shape[3]=0;
      if(m_id!=0)
        ;
        //check that it is not already used 
      else
        ;
        //look for an available id
    }

    Arrayset::~Arrayset() {
      TDEBUG3("Arrayset destructor (id: " << getId() << ")");
      // TODO: What is required here?
    }

    void Arrayset::append( boost::shared_ptr<const Array> array) {
      // TODO: Make a copy of the Array
//      m_array.push_back( array );
      size_t index = m_array.size()-1;
      // TODO: check that id is not already used
      m_index[ array->getId() ] = index;
    }

    void Arrayset::remove( const size_t index) {
      // Get the array using the given index
      boost::shared_ptr<Array> ar = m_array[index];
      // Remove the tuple (id,index) from the map if it exists
      m_index.erase( ar->getId() );
      // Remove the Array from the vector
      if( index<m_array.size() )
        m_array.erase(m_array.begin()+index);
      else
        throw NonExistingElement();
      // Decrease all the index from the map that were above the given
      // on
      std::map<size_t, size_t >::iterator it; 
      for(it = m_index.begin(); it != m_index.end(); ++it)
        if( (*it).second > index )
          --((*it).second);

      // TODO: remove all the relations/members that might be using this 
      // object
    }   

    boost::shared_ptr<const Array> 
    Arrayset::getArray( const size_t index ) const {
      if(!getIsLoaded()) {
        ;//TODO:load
        Arrayset* a=const_cast<Arrayset*>(this);
        a->m_is_loaded = true;
      }
      if( index >= m_array.size() )
        throw IndexError();
      return m_array[index];
    }

    boost::shared_ptr<Array> 
    Arrayset::getArray( const size_t index ) {
      if(!getIsLoaded()) {
        ;//TODO:load
        Arrayset* a=const_cast<Arrayset*>(this);
        a->m_is_loaded = true;
      }
      if( index >= m_array.size() )
        throw IndexError();
      return m_array[index];
    }

    inline const Array& Arrayset::operator[]( const size_t index ) const {
      return *(getArray(index));
    }

    inline Array& Arrayset::operator[]( const size_t index ) {
      return *(getArray(index));
    }

#define REFER_DEF(T,name,D) template<>\
  blitz::Array<T,D> Array::data() \
  { \
    referCheck<D>(); \
    blitz::TinyVector<int,D> shape; \
    boost::shared_ptr<const Arrayset> parent(m_parent_arrayset); \
    parent->getShape(shape); \
    switch(parent->getElementType()) { \
      case name: \
        break; \
      default: \
        error << "Cannot refer to data with a " << \
          "blitz array of a different type." << std::endl; \
        throw TypeError(); \
        break; \
    } \
    return blitz::Array<T,D>(reinterpret_cast<T*>(m_storage), \
        shape, blitz::neverDeleteData); \
  } \

    REFER_DEF(bool,array::t_bool,1)
    REFER_DEF(bool,array::t_bool,2)
    REFER_DEF(bool,array::t_bool,3)
    REFER_DEF(bool,array::t_bool,4)
    REFER_DEF(int8_t,array::t_int8,1)
    REFER_DEF(int8_t,array::t_int8,2)
    REFER_DEF(int8_t,array::t_int8,3)
    REFER_DEF(int8_t,array::t_int8,4)
    REFER_DEF(int16_t,array::t_int16,1)
    REFER_DEF(int16_t,array::t_int16,2)
    REFER_DEF(int16_t,array::t_int16,3)
    REFER_DEF(int16_t,array::t_int16,4)
    REFER_DEF(int32_t,array::t_int32,1)
    REFER_DEF(int32_t,array::t_int32,2)
    REFER_DEF(int32_t,array::t_int32,3)
    REFER_DEF(int32_t,array::t_int32,4)
    REFER_DEF(int64_t,array::t_int64,1)
    REFER_DEF(int64_t,array::t_int64,2)
    REFER_DEF(int64_t,array::t_int64,3)
    REFER_DEF(int64_t,array::t_int64,4)
    REFER_DEF(uint8_t,array::t_uint8,1)
    REFER_DEF(uint8_t,array::t_uint8,2)
    REFER_DEF(uint8_t,array::t_uint8,3)
    REFER_DEF(uint8_t,array::t_uint8,4)
    REFER_DEF(uint16_t,array::t_uint16,1)
    REFER_DEF(uint16_t,array::t_uint16,2)
    REFER_DEF(uint16_t,array::t_uint16,3)
    REFER_DEF(uint16_t,array::t_uint16,4)
    REFER_DEF(uint32_t,array::t_uint32,1)
    REFER_DEF(uint32_t,array::t_uint32,2)
    REFER_DEF(uint32_t,array::t_uint32,3)
    REFER_DEF(uint32_t,array::t_uint32,4)
    REFER_DEF(uint64_t,array::t_uint64,1)
    REFER_DEF(uint64_t,array::t_uint64,2)
    REFER_DEF(uint64_t,array::t_uint64,3)
    REFER_DEF(uint64_t,array::t_uint64,4)
    REFER_DEF(float,array::t_float32,1)
    REFER_DEF(float,array::t_float32,2)
    REFER_DEF(float,array::t_float32,3)
    REFER_DEF(float,array::t_float32,4)
    REFER_DEF(double,array::t_float64,1)
    REFER_DEF(double,array::t_float64,2)
    REFER_DEF(double,array::t_float64,3)
    REFER_DEF(double,array::t_float64,4)
    REFER_DEF(std::complex<float>,array::t_complex64,1)
    REFER_DEF(std::complex<float>,array::t_complex64,2)
    REFER_DEF(std::complex<float>,array::t_complex64,3)
    REFER_DEF(std::complex<float>,array::t_complex64,4)
    REFER_DEF(std::complex<double>,array::t_complex128,1)
    REFER_DEF(std::complex<double>,array::t_complex128,2)
    REFER_DEF(std::complex<double>,array::t_complex128,3)
    REFER_DEF(std::complex<double>,array::t_complex128,4)

  }
}

