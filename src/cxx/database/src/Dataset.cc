/**
 * @file src/cxx/database/src/Dataset.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief Implements the Dataset class.
 */

#include <boost/make_shared.hpp>
#include "database/Dataset.h"
#include "database/XMLParser.h"
#include "database/XMLWriter.h"
#include "database/dataset_common.h"

namespace db = Torch::database;
namespace tdd = Torch::database::detail;

db::Dataset::Dataset(const std::string& name, size_t version) :
  m_name(name),
  m_version(version),
  m_id2arrayset(),
  m_name2relationset()
{
}

db::Dataset::Dataset(const std::string& path) :
  m_name(),
  m_version(0),
  m_id2arrayset(),
  m_name2relationset()
{
  db::detail::XMLParser parser;
  parser.load(path.c_str(), *this, 2); 
}

db::Dataset::Dataset(const db::Dataset& other) :
  m_name(other.m_name),
  m_version(other.m_version),
  m_id2arrayset(other.m_id2arrayset),
  m_name2relationset(other.m_name2relationset)
{
}

db::Dataset::~Dataset() { }

db::Dataset& db::Dataset::operator= (const db::Dataset& other) {
  m_name = other.m_name;
  m_version = other.m_version;
  m_id2arrayset = other.m_id2arrayset;
  m_name2relationset = other.m_name2relationset;
  return *this;
}

/** Operations for accessing the Dataset information **/
const db::Arrayset& db::Dataset::operator[] (size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw db::IndexError(id);
  return *(it->second.get());
}

db::Arrayset& db::Dataset::operator[] (size_t id) {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw db::IndexError(id);
  return *(it->second.get());
}

boost::shared_ptr<const db::Arrayset> db::Dataset::ptr(const size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw db::IndexError(id);
  return it->second;
}

boost::shared_ptr<db::Arrayset> db::Dataset::ptr(const size_t id) {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw db::IndexError(id);
  return it->second;
}

const db::Relationset& db::Dataset::operator[](const std::string& name) const {
  return *ptr(name).get();
}

db::Relationset& db::Dataset::operator[](const std::string& name) {
  return *ptr(name).get();
}

boost::shared_ptr<const db::Relationset> db::Dataset::ptr(const std::string& name) const {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::const_iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw db::NameError(name);
  return it->second;
}

boost::shared_ptr<db::Relationset> db::Dataset::ptr(const std::string& name) {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw db::NameError(name);
  return it->second;
}

size_t db::Dataset::add(boost::shared_ptr<const db::Arrayset> arrayset) {
  return add(*arrayset.get());
}

size_t db::Dataset::add(const db::Arrayset& arrayset) {
  m_id2arrayset[getNextFreeId()] = boost::make_shared<db::Arrayset>(arrayset);
  return m_id2arrayset.rbegin()->first;
}

void db::Dataset::add(size_t id, boost::shared_ptr<const db::Arrayset> arrayset) {
  add(id, *arrayset.get());
}

void db::Dataset::add(size_t id, const db::Arrayset& arrayset) {
  if (m_id2arrayset.find(id) != m_id2arrayset.end()) throw db::IndexError(id);
  m_id2arrayset[id] = boost::make_shared<db::Arrayset>(arrayset);
}

void db::Dataset::set(size_t id, boost::shared_ptr<const db::Arrayset> arrayset) {
  set(id, *arrayset.get());
}

void db::Dataset::set(size_t id, const db::Arrayset& arrayset) {
  if (m_id2arrayset.find(id) == m_id2arrayset.end()) throw db::IndexError(id);
  m_id2arrayset[id] = boost::make_shared<db::Arrayset>(arrayset);
}

void db::Dataset::remove(size_t index) {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.find(index);
  if (it != m_id2arrayset.end()) {
    m_id2arrayset.erase(index);
  }
  else throw db::IndexError(index);
}

size_t db::Dataset::add (const std::string& name, const db::Relationset& relationset) {
  if (m_name2relationset.find(name) != m_name2relationset.end()) throw db::NameError(name);
  m_name2relationset[name] = boost::make_shared<db::Relationset>(relationset);
  m_name2relationset[name]->setParent(this);
  return m_name2relationset.size();
}

size_t db::Dataset::add (const std::string& name, boost::shared_ptr<const db::Relationset> relationset) {
  return add(name, *relationset.get());  
}

void db::Dataset::set (const std::string& name, const db::Relationset& relationset) {
  if (m_name2relationset.find(name) == m_name2relationset.end()) throw db::NameError(name);
  m_name2relationset[name] = boost::make_shared<db::Relationset>(relationset);
}

void db::Dataset::set (const std::string& name, boost::shared_ptr<const db::Relationset> relationset) {
  set(name, *relationset.get());  
}

void db::Dataset::remove (const std::string& name) {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::iterator it = m_name2relationset.find(name);
  if (it != m_name2relationset.end()) {
    it->second->setParent(0); //in case the user holds a copy of this...
    m_name2relationset.erase(name);
  }
  else throw db::NameError(name);
}

size_t db::Dataset::getNextFreeId() const {
  if (!m_id2arrayset.size()) return 1;
  return m_id2arrayset.rbegin()->first + 1; //remember: std::map is sorted by key!
}

void db::Dataset::consolidateIds() {
  size_t id=1;
  for (std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.begin(); it != m_id2arrayset.end(); ++it, ++id) {
    if (id != it->first) { //displaced value, reset
      m_id2arrayset[id] = it->second;
      m_id2arrayset.erase(it->first);
    }
  }
}

void db::Dataset::save(const std::string& path) const {
  tdd::XMLWriter writer;
  writer.write(path.c_str(), *this);
}

bool db::Dataset::exists(size_t arrayset_id) const {
  return (m_id2arrayset.find(arrayset_id) != m_id2arrayset.end());
}

bool db::Dataset::exists(const std::string& relationset_name) const {
  return (m_name2relationset.find(relationset_name) != m_name2relationset.end());
}
