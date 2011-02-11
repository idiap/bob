/**
 * @file src/Relation.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implementation of relations 
 */

#include "database/Relation.h"
#include "database/dataset_common.h"

namespace db = Torch::database;

db::Relation::Relation() :
  m_id(0),
  m_member()
{
}

db::Relation::Relation(const Relation& other) :
  m_id(0),
  m_member(other.m_member)
{
}

db::Relation::~Relation() { }

db::Relation& db::Relation::operator= (const Relation& other) {
  m_id = 0;
  m_member = other.m_member;
  return *this;
}

void db::Relation::remove (size_t index) {
  if (index >= m_member.size()) throw db::IndexError();
  std::list<std::pair<size_t,size_t> >::iterator it = m_member.begin();
  std::advance(it, index);
  m_member.erase(it);
}

const std::pair<size_t, size_t>& db::Relation::operator[] (size_t index) const {
  if (index >= m_member.size()) throw db::IndexError();
  std::list<std::pair<size_t,size_t> >::const_iterator it = m_member.begin();
  std::advance(it, index);
  return *it; 
}
