/**
 * @file src/core/src/Dataset2.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the Dataset class.
 */

#include "core/Dataset2.h"
#include <stdexcept>

namespace Torch {
  namespace core {

    Array::Array(const Arrayset& parent): 
      m_parent_arrayset(parent), m_id(0), m_is_loaded(false), m_filename(""),
      m_loader(l_unknown), m_storage(0) { }

    Array::~Array() {
      std::cout << "Array destructor (id: " << getId() << ")" << std::endl;
      switch(m_parent_arrayset.getArrayType()) {
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

    const Array& Arrayset::operator[]( const size_t id ) const {
      std::map<size_t, boost::shared_ptr<Array> >::const_iterator it = 
        (m_array.find(id));
      if( it == m_array.end() )
        throw IndexError();
      return *(it->second);
    }

    boost::shared_ptr<const Array> 
    Arrayset::getArray( const size_t id ) const {
      std::map<size_t, boost::shared_ptr<Array> >::const_iterator it = 
        (m_array.find(id));
      if( it == m_array.end() )
        throw IndexError();
      return it->second;
    }


    Relationset::Relationset(): 
      m_name("") { }

    Relationset::~Relationset() {
      std::cout << "Relationset destructor (name: " << getName() << ")" << std::endl;
    }

    void Relationset::addRule( boost::shared_ptr<Rule> rule) {
      m_rule.insert( std::pair<std::string,boost::shared_ptr<Rule> >(
        rule->getArraysetRole(), rule) );
    }

    void Relationset::addRelation( boost::shared_ptr<Relation> relation) {
      m_relation.insert( std::pair<size_t,boost::shared_ptr<Relation> >(
        relation->getId(), relation) );
    }

    const Relation& Relationset::operator[]( const size_t id ) const {
      std::map<size_t, boost::shared_ptr<Relation> >::const_iterator it = 
        (m_relation.find(id));
      if( it == m_relation.end() )
        throw IndexError();
      return *(it->second);
    }

    boost::shared_ptr<const Relation> 
    Relationset::getRelation( const size_t id ) const {
      std::map<size_t, boost::shared_ptr<Relation> >::const_iterator it = 
        (m_relation.find(id));
      if( it == m_relation.end() )
        throw IndexError();
      return it->second;
    }


    Rule::Rule(): 
      m_arraysetrole(""), m_min(1), m_max(1) { }

    Rule::~Rule() {
      std::cout << "Rule destructor (Arrayset-role: " << getArraysetRole() << ")" << std::endl;
    }


    Relation::Relation(): 
      m_id(0) { }

    Relation::~Relation() {
      std::cout << "Relation destructor (id: " << getId() << ")" << std::endl;
    }

    void Relation::addMember( boost::shared_ptr<Member> member) {
      size_t_pair ids( member->getArrayId(), member->getArraysetId());
      m_member.insert( std::pair<size_t_pair,boost::shared_ptr<Member> >(
        ids, member) );
    }


    Member::Member(): 
      m_array_id(0), m_arrayset_id(0) { }

    Member::~Member() {
      std::cout << "Member destructor (id: " << getArrayId() << "-" << getArraysetId() << ")" << std::endl;
    }


    Dataset::Dataset() { }

    Dataset::~Dataset() {
      std::cout << "Dataset destructor" << std::endl;
    }

    void Dataset::addArrayset( boost::shared_ptr<Arrayset> arrayset) {
      m_arrayset.insert( std::pair<size_t,boost::shared_ptr<Arrayset> >(
        arrayset->getId(), arrayset) );
    }

    void Dataset::addRelationset( boost::shared_ptr<Relationset> relationset) {
      m_relationset.insert( std::pair<std::string,boost::shared_ptr<Relationset> >(
        relationset->getName(), relationset) );
    }

    const Arrayset& Dataset::operator[]( const size_t id ) const {
      std::map<size_t, boost::shared_ptr<Arrayset> >::const_iterator it = 
        (m_arrayset.find(id));
      if( it == m_arrayset.end() )
        throw IndexError();
      return *(it->second);
    }

    boost::shared_ptr<const Arrayset> 
    Dataset::getArrayset( const size_t id ) const {
      std::map<size_t, boost::shared_ptr<Arrayset> >::const_iterator it = 
        (m_arrayset.find(id));
      if( it == m_arrayset.end() )
        throw IndexError();
      return it->second;
    }

    const Relationset& Dataset::operator[]( const std::string& name ) const {
      std::map<std::string, boost::shared_ptr<Relationset> >::const_iterator 
        it = (m_relationset.find(name));
      if( it == m_relationset.end() )
        throw IndexError();
      return *(it->second);
    }

    boost::shared_ptr<const Relationset> 
    Dataset::getRelationset( const std::string& name ) const {
      std::map<std::string, boost::shared_ptr<Relationset> >::const_iterator it = 
        (m_relationset.find(name));
      if( it == m_relationset.end() )
        throw IndexError();
      return it->second;
    }


  }
}

