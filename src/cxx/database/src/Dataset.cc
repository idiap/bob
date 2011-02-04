/**
 * @file src/cxx/database/src/Dataset.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief Implements the Dataset class.
 */

#include "database/Dataset.h"
#include "database/XMLParser.h"
#include "database/dataset_common.h"
#include "database/Arrayset.h"
#include "database/Relationset.h"

namespace db = Torch::database;

db::Dataset::Dataset(const std::string& name, size_t version)
  : m_name(name),
    m_version(version),
    m_id2arrayset(),
    m_name2relationset()
{
}

db::Dataset::Dataset(const std::string& path)
  : m_name(),
    m_version(0),
    m_id2arrayset(),
    m_name2relationset()
{
  //fill up using the parser
  db::XMLParser parser;
  parser.load(path.c_str(), *this, 2); 
}

db::Dataset::~Dataset() { }

/** Operations for accessing the Dataset information **/
const db::Arrayset& db::Dataset::operator[] (size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw IndexError();
  return *(it->second.get());
}

db::Arrayset& db::Dataset::operator[] (size_t id) {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw IndexError();
  return *(it->second.get());
}

boost::shared_ptr<const db::Arrayset> 
db::Dataset::getArrayset(const size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw IndexError();
  return it->second;
}

boost::shared_ptr<db::Arrayset> db::Dataset::getArrayset(const size_t id) {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw IndexError();
  return it->second;
}

const db::Relationset& db::Dataset::operator[](const std::string& name) const {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::const_iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw IndexError();
  return *(it->second.get());
}

db::Relationset& db::Dataset::operator[](const std::string& name) {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw IndexError();
  return *(it->second.get());
}

boost::shared_ptr<const db::Relationset> 
db::Dataset::getRelationset(const std::string& name) const {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::const_iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw IndexError();
  return it->second;
}

boost::shared_ptr<db::Relationset> 
db::Dataset::getRelationset(const std::string& name) {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw IndexError();
  return it->second;
}

/** Operations dealing with modification of the Dataset **/
void db::Dataset::add(boost::shared_ptr<db::Arrayset> arrayset) {
  m_id2arrayset[arrayset->getId()] = arrayset;
}

void db::Dataset::add(const db::Arrayset& arrayset) {
  m_id2arrayset[arrayset.getId()] = 
    boost::shared_ptr<db::Arrayset>(new db::Arrayset(arrayset));
}

void db::Dataset::remove(size_t index) {
  m_id2arrayset.erase(index);
}

void db::Dataset::add (boost::shared_ptr<db::Relationset> relationset) {
  m_name2relationset[relationset->getName()] = relationset;
}

void db::Dataset::add (const db::Relationset& arrayset) {
  m_name2relationset[relationset.getName()] = 
    boost::shared_ptr<db::Arrayset>(new db::Relationset(relationset));
}

void db::Dataset::remove (const std::string& name) {
  m_name2relationset.erase(name);
}
