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

namespace db = Torch::database;
namespace tdd = Torch::database::detail;

namespace Torch { namespace database { namespace detail {

  /**
   * A small predicate class to help with Id comparison for the m_array
   */
  struct equal_arrayset_id {
    size_t id;

    equal_arrayset_id(size_t id) : id(id) { }

    inline bool operator() (const boost::shared_ptr<db::Arrayset>& v)
    { return v->getId() == id; }

  };

  /**
   * Another predicate to help list sorting
   */
  static bool arrayset_is_smaller (const boost::shared_ptr<db::Arrayset>& v1,
                   const boost::shared_ptr<db::Arrayset>& v2) {
    return v1->getId() < v2->getId();
  }

} } }

db::Dataset::Dataset(const std::string& name, size_t version) :
  m_name(name),
  m_version(version),
  m_arrayset(),
  m_id2arrayset()
  //m_name2relationset()
{
}

db::Dataset::Dataset(const std::string& path) :
  m_name(),
  m_version(0),
  m_arrayset(),
  m_id2arrayset()
  //m_name2relationset()
{
  //LES: Please fill up using the parser
  //db::XMLParser parser;
  //parser.load(path.c_str(), *this, 2); 
  tdd::XMLParser parser;
  parser.load(path.c_str(), *this, 2);
}

db::Dataset::Dataset(const db::Dataset& other) :
  m_name(other.m_name),
  m_version(other.m_version),
  m_arrayset(other.m_arrayset),
  m_id2arrayset(other.m_id2arrayset)
  //m_name2relationset(other.m_name2relationset)
{
}

db::Dataset::~Dataset() { }

db::Dataset& db::Dataset::operator= (const db::Dataset& other) {
  m_name = other.m_name;
  m_version = other.m_version;
  m_arrayset = other.m_arrayset;
  m_id2arrayset = other.m_id2arrayset;
  //m_name2relationset = other.m_name2relationset;
  return *this;
}

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

boost::shared_ptr<const db::Arrayset> db::Dataset::ptr(const size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::const_iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw IndexError();
  return it->second;
}

boost::shared_ptr<db::Arrayset> db::Dataset::ptr(const size_t id) {
  std::map<size_t, boost::shared_ptr<db::Arrayset> >::iterator it = m_id2arrayset.find(id);
  if (it == m_id2arrayset.end()) throw IndexError();
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
  if (it == m_name2relationset.end()) throw IndexError();
  return it->second;
}

boost::shared_ptr<db::Relationset> db::Dataset::ptr(const std::string& name) {
  std::map<std::string, boost::shared_ptr<db::Relationset> >::iterator it = m_name2relationset.find(name);
  if (it == m_name2relationset.end()) throw IndexError();
  return it->second;
}

size_t db::Dataset::add(boost::shared_ptr<const db::Arrayset> arrayset) {
  return add(*arrayset.get());
}

size_t db::Dataset::add(const db::Arrayset& arrayset) {
  size_t use_id = arrayset.getId();
  if (!use_id) use_id = getNextFreeId();
  boost::shared_ptr<db::Arrayset> acopy(new db::Arrayset(arrayset));
  acopy->setId(use_id);
  m_id2arrayset[use_id] = acopy;
  m_arrayset.push_back(acopy);
  return use_id;
}

void db::Dataset::remove(size_t index) {
  m_id2arrayset.erase(index);
  m_arrayset.remove_if(tdd::equal_arrayset_id(index));
}

size_t db::Dataset::add (const db::Relationset& relationset) {
  m_name2relationset[relationset.getName()] = 
    boost::shared_ptr<db::Relationset>(new db::Relationset(relationset));
  return m_name2relationset.size();
}

size_t db::Dataset::add (boost::shared_ptr<const db::Relationset> relationset) {
  return add(*relationset.get());  
}

void db::Dataset::remove (const std::string& name) {
  m_name2relationset.erase(name);
}

size_t db::Dataset::getNextFreeId() const {
  if (!m_arrayset.size()) return 1;
  return (*std::max_element(m_arrayset.begin(), m_arrayset.end(), tdd::arrayset_is_smaller))->getId() + 1;
}

void db::Dataset::consolidateIds() {
  m_arrayset.sort(tdd::arrayset_is_smaller);
  m_id2arrayset.clear();
  size_t id=1;
  for (std::list<boost::shared_ptr<db::Arrayset> >::iterator it = m_arrayset.begin(); it != m_arrayset.end(); ++it, ++id) {
    m_id2arrayset[id] = *it;
  }
}
