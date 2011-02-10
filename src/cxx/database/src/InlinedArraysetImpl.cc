/**
 * @file src/InlinedArraysetImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the InlinedArraysetImpl type
 */

#include <algorithm>

#include "database/InlinedArraysetImpl.h"
#include "database/dataset_common.h"

namespace db = Torch::database;
namespace tdd = Torch::database::detail;

namespace Torch { namespace database { namespace detail {

  /**
   * A small predicate class to help with Id comparison for the m_array
   */
  struct equal_array_id {
    size_t id;

    equal_array_id(size_t id) : id(id) { }

    inline bool operator() (const boost::shared_ptr<db::Array>& v)
    { return v->getId() == id; }

  };

  /**
   * Another predicate to help list sorting
   */
  static bool array_is_smaller (const boost::shared_ptr<db::Array>& v1,
                   const boost::shared_ptr<db::Array>& v2) {
    return v1->getId() < v2->getId();
  }

} } }

tdd::InlinedArraysetImpl::InlinedArraysetImpl()
  : m_elementtype(Torch::core::array::t_unknown),
    m_ndim(0),
    m_array(),
    m_index()
{
}

tdd::InlinedArraysetImpl::InlinedArraysetImpl
  (const tdd::InlinedArraysetImpl& other)
  : m_elementtype(other.m_elementtype),
    m_ndim(other.m_ndim),
    m_array(other.m_array),
    m_index(other.m_index)
{
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
}

tdd::InlinedArraysetImpl::~InlinedArraysetImpl() { }

tdd::InlinedArraysetImpl& tdd::InlinedArraysetImpl::operator= 
(const InlinedArraysetImpl& other) {
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  m_array = other.m_array;
  m_index = other.m_index;
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

const db::Array& tdd::InlinedArraysetImpl::operator[] (size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Array> >::const_iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError();
  return *(it->second.get());
}

db::Array& tdd::InlinedArraysetImpl::operator[] (size_t id) {
  std::map<size_t, boost::shared_ptr<db::Array> >::iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError();
  return *(it->second.get());
}

boost::shared_ptr<const db::Array> tdd::InlinedArraysetImpl::ptr
(size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Array> >::const_iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError();
  return it->second;
}

boost::shared_ptr<db::Array> tdd::InlinedArraysetImpl::ptr (size_t id) {
  std::map<size_t, boost::shared_ptr<db::Array> >::iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError();
  return it->second;
}

void tdd::InlinedArraysetImpl::checkCompatibility (const db::Array& array) const {
  if (m_elementtype != Torch::core::array::t_unknown) {
    if (array.getElementType() != m_elementtype) throw db::TypeError();
    if (array.getNDim() != m_ndim) throw db::DimensionError();
    for (size_t i=0; i<m_ndim; ++i)
      if (array.getShape()[i] != m_shape[i]) throw db::DimensionError();
  }
}

void tdd::InlinedArraysetImpl::updateTyping (const db::Array& array) {
  if (m_elementtype == Torch::core::array::t_unknown) {
    m_elementtype = array.getElementType();
    m_ndim = array.getNDim();
    for (size_t i=0; i<m_ndim; ++i) m_shape[i] = array.getShape()[i];
  }
}

size_t tdd::InlinedArraysetImpl::add (boost::shared_ptr<const db::Array> array)
{ 
  return add(*array.get()); 
}

size_t tdd::InlinedArraysetImpl::add(const db::Array& array) {
  checkCompatibility(array);

  size_t use_id = array.getId();
  if (!use_id) use_id = getNextFreeId();
  boost::shared_ptr<db::Array> acopy(new db::Array(array));
  acopy->setId(use_id);
  m_index[use_id] = acopy;
  m_array.push_back(acopy);

  updateTyping(array);
  return use_id;
}

size_t tdd::InlinedArraysetImpl::adopt (boost::shared_ptr<db::Array> array) {
  checkCompatibility(*array.get());
  
  size_t use_id = array->getId();
  if (!use_id) use_id = getNextFreeId();
  array->setId(use_id);
  m_index[use_id] = array;
  m_array.push_back(array);
  
  updateTyping(*array.get());
  return use_id;
}

void tdd::InlinedArraysetImpl::remove(size_t id) {
  m_index.erase(id);
  m_array.remove_if(tdd::equal_array_id(id));
  if (m_array.size() == 0) { //uninitialize
    m_elementtype = Torch::core::array::t_unknown;
    m_ndim = 0;
  }
}

void tdd::InlinedArraysetImpl::remove(boost::shared_ptr<const db::Array> array) {
  remove(array->getId());
}

void tdd::InlinedArraysetImpl::remove(const db::Array& array) {
  remove(array.getId());
}

size_t tdd::InlinedArraysetImpl::getNextFreeId() const {
  if (!m_array.size()) return 1;
  return (*std::max_element(m_array.begin(), m_array.end(), tdd::array_is_smaller))->getId() + 1;
}

void tdd::InlinedArraysetImpl::consolidateIds() {
  m_array.sort(tdd::array_is_smaller);
  m_index.clear();
  size_t id=1;
  for (std::list<boost::shared_ptr<db::Array> >::iterator it = m_array.begin();
       it != m_array.end(); ++it, ++id) {
    m_index[id] = *it;
  }
}
