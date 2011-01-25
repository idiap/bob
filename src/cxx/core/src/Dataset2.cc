/**
 * @file src/cxx/core/src/Dataset2.cc
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
      TDEBUG3("Array destructor (id: " << getId() << ")");
      switch(m_parent_arrayset.getElementType()) {
        case array::t_bool:
          delete [] static_cast<bool*>(m_storage); break;
        case array::t_int8:
          delete [] static_cast<int8_t*>(m_storage); break;
        case array::t_int16:
          delete [] static_cast<int16_t*>(m_storage); break;
        case array::t_int32:
          delete [] static_cast<int32_t*>(m_storage); break;
        case array::t_int64:
          delete [] static_cast<int64_t*>(m_storage); break;
        case array::t_uint8:
          delete [] static_cast<uint8_t*>(m_storage); break;
        case array::t_uint16:
          delete [] static_cast<uint16_t*>(m_storage); break;
        case array::t_uint32:
          delete [] static_cast<uint32_t*>(m_storage); break;
        case array::t_uint64:
          delete [] static_cast<uint64_t*>(m_storage); break;
        case array::t_float32:
          delete [] static_cast<float*>(m_storage); break;
        case array::t_float64:
          delete [] static_cast<double*>(m_storage); break;
        case array::t_float128:
          delete [] static_cast<long double*>(m_storage); break;
        case array::t_complex64:
          delete [] static_cast<std::complex<float>* >(m_storage); break;
        case array::t_complex128:
          delete [] static_cast<std::complex<double>* >(m_storage); break;
        case array::t_complex256:
          delete [] static_cast<std::complex<long double>* >(m_storage); 
          break;
        default:
          break;
      }
    } 


    Arrayset::Arrayset(): 
      m_id(0), m_n_dim(0), m_n_elem(0), m_element_type(array::t_unknown), 
      m_role(""), m_is_loaded(false), m_filename(""), m_loader(l_unknown) 
    {
      m_shape[0]=m_shape[1]=m_shape[2]=m_shape[3]=0; 
    }

    Arrayset::~Arrayset() {
      TDEBUG3("Arrayset destructor (id: " << getId() << ")");
      for(iterator it=begin(); it!=end(); ++it)
        it->second.reset();
    }

    void Arrayset::append( boost::shared_ptr<Array> array) {
      m_array.insert( std::pair<size_t,boost::shared_ptr<Array> >(
        array->getId(), array) );
    }

    const Array& Arrayset::operator[]( const size_t id ) const {
      if(!getIsLoaded()) {
        ;//TODO:load
        Arrayset* a=const_cast<Arrayset*>(this);
        a->setIsLoaded(true);
      }
      std::map<size_t, boost::shared_ptr<Array> >::const_iterator it = 
        (m_array.find(id));
      if( it == m_array.end() )
        throw IndexError();
      return *(it->second);
    }

    boost::shared_ptr<const Array> 
    Arrayset::getArray( const size_t id ) const {
      if(!getIsLoaded()) {
        ;//TODO:load
        Arrayset* a=const_cast<Arrayset*>(this);
        a->setIsLoaded(true);
      }
      std::map<size_t, boost::shared_ptr<Array> >::const_iterator it = 
        (m_array.find(id));
      if( it == m_array.end() )
        throw IndexError();
      return it->second;
    }

    boost::shared_ptr<Array> 
    Arrayset::getArray( const size_t id ) {
      if(!getIsLoaded()) {
        ;//TODO:load
        Arrayset* a=const_cast<Arrayset*>(this);
        a->setIsLoaded(true);
      }
      std::map<size_t, boost::shared_ptr<Array> >::iterator it = 
        (m_array.find(id));
      if( it == m_array.end() )
        throw IndexError();
      return it->second;
    }

    Relationset::Relationset(): 
      m_name("") { }

    Relationset::~Relationset() {
      TDEBUG3("Relationset destructor (name: " << getName() << ")");
    }

    void Relationset::append( boost::shared_ptr<Rule> rule) {
      m_rule.insert( std::pair<std::string,boost::shared_ptr<Rule> >(
        rule->getArraysetRole(), rule) );
    }

    void Relationset::append( boost::shared_ptr<Relation> relation) {
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

    boost::shared_ptr<Relation> 
    Relationset::getRelation( const size_t id ) {
      std::map<size_t, boost::shared_ptr<Relation> >::iterator it = 
        (m_relation.find(id));
      if( it == m_relation.end() )
        throw IndexError();
      return it->second;
    }

    const Rule& Relationset::operator[]( const std::string& role ) const {
      std::map<std::string, boost::shared_ptr<Rule> >::const_iterator it = 
        (m_rule.find(role));
      if( it == m_rule.end() )
        throw IndexError();
      return *(it->second);
    }

    boost::shared_ptr<const Rule> Relationset::getRule( const std::string& role ) const {
      std::map<std::string, boost::shared_ptr<Rule> >::const_iterator it = (m_rule.find(role));
      if( it == m_rule.end() ) throw IndexError();
      return it->second;
    }

    boost::shared_ptr<Rule> Relationset::getRule( const std::string& role ) {
      std::map<std::string, boost::shared_ptr<Rule> >::iterator it = (m_rule.find(role));
      if( it == m_rule.end() ) throw IndexError();
      return it->second;
    }

    Rule::Rule(): 
      m_arraysetrole(""), m_min(1), m_max(1) { }

    Rule::~Rule() {
      TDEBUG3("Rule destructor (Arrayset-role: " << getArraysetRole() << ")");
    }


    Relation::Relation( boost::shared_ptr<std::map<size_t,std::string> > 
      id_role): m_id(0), m_id_role(id_role)
    { 
    }

    Relation::~Relation() {
      TDEBUG3("Relation destructor (id: " << getId() << ")");
    }

    void Relation::append( boost::shared_ptr<Member> member) {
      size_t_pair ids( member->getArrayId(), member->getArraysetId());
      m_member.insert( std::pair<size_t_pair,boost::shared_ptr<Member> >(
        ids, member) );
    }


    Member::Member(): 
      m_array_id(0), m_arrayset_id(0) { }

    Member::~Member() {
      TDEBUG3("Member destructor (id: " << getArrayId() << "-" << 
        getArraysetId() << ")");
    }


    Dataset::Dataset() { }

    Dataset::~Dataset() {
      TDEBUG3("Dataset destructor");
    }

    void Dataset::append( boost::shared_ptr<Arrayset> arrayset) {
      m_arrayset.insert( std::pair<size_t,boost::shared_ptr<Arrayset> >(
        arrayset->getId(), arrayset) );
    }

    void Dataset::append( boost::shared_ptr<Relationset> relationset) {
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

    boost::shared_ptr<Arrayset>
    Dataset::getArrayset( const size_t id ) {
      std::map<size_t, boost::shared_ptr<Arrayset> >::iterator it = 
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

    boost::shared_ptr<Relationset> 
    Dataset::getRelationset( const std::string& name ) {
      std::map<std::string, boost::shared_ptr<Relationset> >::iterator it = 
        (m_relationset.find(name));
      if( it == m_relationset.end() )
        throw IndexError();
      return it->second;
    }

  }
}
