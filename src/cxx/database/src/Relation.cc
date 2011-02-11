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
  //m_parent(),
  m_id(0),
  m_member()
{
}

db::Relation::Relation(const Relation& other) :
  //m_parent(),
  m_id(0),
  m_member(other.m_member)
{
}

db::Relation::~Relation() { }

db::Relation& db::Relation::operator= (const Relation& other) {
  //m_parent.reset();
  m_id = 0;
  m_member = other.m_member;
  return *this;
}

void db::Relation::add (const std::string& role, size_t arraysetid) {
  if (!arraysetid) throw db::IndexError();
  m_member[role] = std::make_pair(arraysetid, 0);
}

void db::Relation::add (const std::string& role, size_t arraysetid, size_t arrayid) {
  if (!arraysetid) throw db::IndexError();
  m_member[role] = std::make_pair(arraysetid, arrayid);
}

void db::Relation::remove (const std::string& role) {
  m_member.erase(role);
}

const std::pair<size_t, size_t>& db::Relation::operator[] (const std::string& role) {
  std::map<std::string, std::pair<size_t, size_t> >::const_iterator it = m_member.find(role);
  if (it == m_member.end()) throw IndexError();
  return it->second;
}
