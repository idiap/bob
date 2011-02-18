/**
 * @file src/InlinedArraysetImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the InlinedArraysetImpl type
 */

#include <algorithm>
#include <boost/make_shared.hpp>

#include "database/InlinedArraysetImpl.h"
#include "database/dataset_common.h"

namespace db = Torch::database;
namespace tdd = Torch::database::detail;

tdd::InlinedArraysetImpl::InlinedArraysetImpl()
  : m_elementtype(Torch::core::array::t_unknown),
    m_ndim(0),
    m_index()
{
}

tdd::InlinedArraysetImpl::InlinedArraysetImpl
  (const tdd::InlinedArraysetImpl& other)
  : m_elementtype(other.m_elementtype),
    m_ndim(other.m_ndim),
    m_index(other.m_index)
{
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
}

tdd::InlinedArraysetImpl::~InlinedArraysetImpl() { }

tdd::InlinedArraysetImpl& tdd::InlinedArraysetImpl::operator= 
(const InlinedArraysetImpl& other) {
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  m_index = other.m_index;
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

const db::Array& tdd::InlinedArraysetImpl::operator[] (size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Array> >::const_iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError(id);
  return *(it->second.get());
}

db::Array& tdd::InlinedArraysetImpl::operator[] (size_t id) {
  std::map<size_t, boost::shared_ptr<db::Array> >::iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError(id);
  return *(it->second.get());
}

boost::shared_ptr<const db::Array> tdd::InlinedArraysetImpl::ptr
(size_t id) const {
  std::map<size_t, boost::shared_ptr<db::Array> >::const_iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError(id);
  return it->second;
}

boost::shared_ptr<db::Array> tdd::InlinedArraysetImpl::ptr (size_t id) {
  std::map<size_t, boost::shared_ptr<db::Array> >::iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw db::IndexError(id);
  return it->second;
}

void tdd::InlinedArraysetImpl::checkCompatibility (const db::Array& array) const {
  if (m_elementtype != Torch::core::array::t_unknown) {
    if (array.getElementType() != m_elementtype) throw db::TypeError(array.getElementType(), m_elementtype);
    if (array.getNDim() != m_ndim) throw db::DimensionError(array.getNDim(), m_ndim);
    for (size_t i=0; i<m_ndim; ++i)
      if (array.getShape()[i] != m_shape[i]) throw db::DimensionError(array.getShape()[i], m_shape[i]);
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
  updateTyping(array);
  m_index[getNextFreeId()] = boost::make_shared<db::Array>(array); 
  return m_index.rbegin()->first;
}

void tdd::InlinedArraysetImpl::add(size_t id, boost::shared_ptr<const Torch::database::Array> array) {
  return add(id, *array.get());
}

void tdd::InlinedArraysetImpl::add(size_t id, const Torch::database::Array& array) {
  if (m_index.find(id) != m_index.end()) throw db::IndexError(id);
  checkCompatibility(array);
  updateTyping(array);
  m_index[id] = boost::make_shared<db::Array>(array); 
}

size_t tdd::InlinedArraysetImpl::adopt (boost::shared_ptr<db::Array> array) {
  checkCompatibility(*array.get());
  updateTyping(*array.get());
  m_index[getNextFreeId()] = array;
  return m_index.rbegin()->first;
}

void tdd::InlinedArraysetImpl::remove(size_t id) {
  if (m_index.find(id) == m_index.end()) throw db::IndexError(id);
  m_index.erase(id);
  if (m_index.size() == 0) { //uninitialize
    m_elementtype = Torch::core::array::t_unknown;
    m_ndim = 0;
  }
}

size_t tdd::InlinedArraysetImpl::getNextFreeId() const {
  if (!m_index.size()) return 1;
  return m_index.rbegin()->first + 1; //remember: std::map is sorted by key!
}

void tdd::InlinedArraysetImpl::consolidateIds() {
  std::vector<size_t> keys;
  for (std::map<size_t, boost::shared_ptr<db::Array> >::iterator 
      it = m_index.begin(); it != m_index.end(); ++it)
    keys.push_back(it->first);

  for (size_t id=1; id<=m_index.size(); ++id)
    if (id != keys[id-1]) { //displaced value, reset
      m_index[id] = m_index[keys[id-1]];
      m_index.erase(keys[id-1]);
    }
}

bool tdd::InlinedArraysetImpl::exists(size_t id) const {
  return (m_index.find(id) != m_index.end());
}
