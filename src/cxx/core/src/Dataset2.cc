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
      m_arrayset.push_back( arrayset);
      m_arrayset_index.insert( std::pair<size_t,size_t>(m_arrayset.size()-1,
        arrayset->getId() ) );
    }

    void Dataset::append( boost::shared_ptr<Relationset> relationset) {
      m_relationset.insert( std::pair<std::string,boost::shared_ptr<Relationset> >(
        relationset->getName(), relationset) );
    }

    const Arrayset& Dataset::operator[]( const size_t index ) const {
      if( index >= m_arrayset.size() )
        throw IndexError();
      else
        return *(m_arrayset[index]);
    }

    Arrayset& Dataset::operator[]( const size_t index ) {
      if( index >= m_arrayset.size() )
        throw IndexError();
      else
        return *(m_arrayset[index]);
    }

    boost::shared_ptr<const Arrayset> 
    Dataset::getArrayset( const size_t index ) const {
      if( index >= m_arrayset.size() )
        throw IndexError();
      else
        return m_arrayset[index];
    }

    boost::shared_ptr<Arrayset>
    Dataset::getArrayset( const size_t index ) {
      if( index >= m_arrayset.size() )
        throw IndexError();
      else
        return m_arrayset[index];
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
