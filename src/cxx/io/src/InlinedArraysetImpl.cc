/**
 * @file src/InlinedArraysetImpl.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the InlinedArraysetImpl type
 */

#include <algorithm>
#include <boost/make_shared.hpp>

#include "io/InlinedArraysetImpl.h"

namespace io = Torch::io;
namespace iod = io::detail;

iod::InlinedArraysetImpl::InlinedArraysetImpl()
  : m_elementtype(Torch::core::array::t_unknown),
    m_ndim(0),
    m_index()
{
}

iod::InlinedArraysetImpl::InlinedArraysetImpl
  (const iod::InlinedArraysetImpl& other)
  : m_elementtype(other.m_elementtype),
    m_ndim(other.m_ndim),
    m_index(other.m_index)
{
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
}

iod::InlinedArraysetImpl::~InlinedArraysetImpl() { }

iod::InlinedArraysetImpl& iod::InlinedArraysetImpl::operator= 
(const InlinedArraysetImpl& other) {
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  m_index = other.m_index;
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

const io::Array& iod::InlinedArraysetImpl::operator[] (size_t id) const {
  std::map<size_t, boost::shared_ptr<io::Array> >::const_iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw io::IndexError(id);
  return *(it->second.get());
}

io::Array& iod::InlinedArraysetImpl::operator[] (size_t id) {
  std::map<size_t, boost::shared_ptr<io::Array> >::iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw io::IndexError(id);
  return *(it->second.get());
}

boost::shared_ptr<const io::Array> iod::InlinedArraysetImpl::ptr
(size_t id) const {
  std::map<size_t, boost::shared_ptr<io::Array> >::const_iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw io::IndexError(id);
  return it->second;
}

boost::shared_ptr<io::Array> iod::InlinedArraysetImpl::ptr (size_t id) {
  std::map<size_t, boost::shared_ptr<io::Array> >::iterator it = 
    m_index.find(id);
  if (it == m_index.end()) throw io::IndexError(id);
  return it->second;
}

void iod::InlinedArraysetImpl::checkCompatibility (const io::Array& array) const {
  if (m_elementtype != Torch::core::array::t_unknown) {
    if (array.getElementType() != m_elementtype) throw io::TypeError(array.getElementType(), m_elementtype);
    if (array.getNDim() != m_ndim) throw io::DimensionError(array.getNDim(), m_ndim);
    for (size_t i=0; i<m_ndim; ++i)
      if (array.getShape()[i] != m_shape[i]) throw io::DimensionError(array.getShape()[i], m_shape[i]);
  }
}

void iod::InlinedArraysetImpl::updateTyping (const io::Array& array) {
  if (m_elementtype == Torch::core::array::t_unknown) {
    m_elementtype = array.getElementType();
    m_ndim = array.getNDim();
    for (size_t i=0; i<m_ndim; ++i) m_shape[i] = array.getShape()[i];
  }
}

size_t iod::InlinedArraysetImpl::add (boost::shared_ptr<const io::Array> array)
{ 
  return add(*array.get()); 
}

size_t iod::InlinedArraysetImpl::add(const io::Array& array) {
  checkCompatibility(array);
  updateTyping(array);
  m_index[getNextFreeId()] = boost::make_shared<io::Array>(array); 
  return m_index.rbegin()->first;
}

void iod::InlinedArraysetImpl::add(size_t id, boost::shared_ptr<const io::Array> array) {
  return add(id, *array.get());
}

void iod::InlinedArraysetImpl::add(size_t id, const io::Array& array) {
  if (m_index.find(id) != m_index.end()) throw io::IndexError(id);
  checkCompatibility(array);
  updateTyping(array);
  m_index[id] = boost::make_shared<io::Array>(array); 
}

size_t iod::InlinedArraysetImpl::adopt (boost::shared_ptr<io::Array> array) {
  checkCompatibility(*array.get());
  updateTyping(*array.get());
  m_index[getNextFreeId()] = array;
  return m_index.rbegin()->first;
}

void iod::InlinedArraysetImpl::remove(size_t id) {
  if (m_index.find(id) == m_index.end()) throw io::IndexError(id);
  m_index.erase(id);
  if (m_index.size() == 0) { //uninitialize
    m_elementtype = Torch::core::array::t_unknown;
    m_ndim = 0;
  }
}

size_t iod::InlinedArraysetImpl::getNextFreeId() const {
  if (!m_index.size()) return 1;
  return m_index.rbegin()->first + 1; //remember: std::map is sorted by key!
}

void iod::InlinedArraysetImpl::consolidateIds() {
  std::vector<size_t> keys;
  for (std::map<size_t, boost::shared_ptr<io::Array> >::iterator 
      it = m_index.begin(); it != m_index.end(); ++it)
    keys.push_back(it->first);

  for (size_t id=1; id<=m_index.size(); ++id)
    if (id != keys[id-1]) { //displaced value, reset
      m_index[id] = m_index[keys[id-1]];
      m_index.erase(keys[id-1]);
    }
}

bool iod::InlinedArraysetImpl::exists(size_t id) const {
  return (m_index.find(id) != m_index.end());
}
