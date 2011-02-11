/**
 * @file database/src/Relationset.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implementation of the Relationset class for Torch databases
 */

#include <list>
#include "database/Relationset.h"
#include "database/dataset_common.h"

namespace db = Torch::database;

db::Relationset::Relationset () :
  m_name(),
  m_relation()
{
}

db::Relationset::Relationset (const Relationset& other) :
  m_name(other.m_name),
  m_relation(other.m_relation)
{
}

db::Relationset::~Relationset() { }

db::Relationset& db::Relationset::operator= (const db::Relationset& other) {
  m_name = other.m_name;
  m_relation = other.m_relation;
  return *this;
}

size_t db::Relationset::getNextFreeId() const {
  if (!m_relation.size()) return 1;
  size_t max = 0;
  for (std::map<size_t, boost::shared_ptr<db::Relation> >::const_iterator it = m_relation.begin();
      it != m_relation.end(); ++it) {
    if (it->first > max) max = it->first;
  }
  return max + 1;
}

void db::Relationset::consolidateIds() {
  std::list<boost::shared_ptr<db::Relation> > l;
  for (std::map<size_t, boost::shared_ptr<db::Relation> >::iterator it = m_relation.begin(); it != m_relation.end(); ++it) l.push_back(it->second);
  m_relation.clear();
  size_t id=1;
  for (std::list<boost::shared_ptr<db::Relation> >::iterator it = l.begin();
       it != l.end(); ++it, ++id) {
    (*it)->setId(id);
    m_relation[id] = *it;
  }
}

size_t db::Relationset::add (const db::Relation& relation) {
  size_t use_id = relation.getId();
  if (!use_id) use_id = getNextFreeId();
  boost::shared_ptr<db::Relation> rcopy(new db::Relation(relation));
  rcopy->setId(use_id);
  m_relation[use_id] = rcopy;
  return use_id;
}

size_t db::Relationset::add (boost::shared_ptr<const db::Relation> relation) {
  return add(*relation.get());
}

void db::Relationset::remove (size_t id) {
  m_relation.erase(id);
}

const db::Relation& db::Relationset::operator[] (size_t id) const {
  return *ptr(id).get();
}

boost::shared_ptr<const db::Relation> db::Relationset::ptr (size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Relation> >::const_iterator it = m_relation.find(id);
  if (it == m_relation.end()) throw db::IndexError();
  return it->second;
}
      
size_t db::Relationset::add (const db::Rule& rule) {
  m_rule[rule.getRole()] = boost::shared_ptr<db::Rule>(new db::Rule(rule));
  return m_rule.size();
}

size_t db::Relationset::add (boost::shared_ptr<const db::Rule> rule) {
  return add(*rule.get()); 
}

void db::Relationset::remove(const std::string& rulerole) {
  m_rule.erase(rulerole);
}

boost::shared_ptr<const db::Rule> db::Relationset::ptr (const std::string& role) const {
  std::map<std::string, boost::shared_ptr<db::Rule> >::const_iterator it = m_rule.find(role);
  if (it == m_rule.end()) throw db::IndexError();
  return it->second;
}
      
const db::Rule& db::Relationset::operator[] (const std::string& role) const {
  return *ptr(role).get();
}

