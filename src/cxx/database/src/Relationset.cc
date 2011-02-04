/**
 * @file Relationset.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Defines the Relationset implementation  
 */

#include "database/Relationset.h"

namespace db = Torch::database

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

