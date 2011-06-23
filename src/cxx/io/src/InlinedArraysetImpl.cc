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
    m_data()
{
}

iod::InlinedArraysetImpl::InlinedArraysetImpl
  (const iod::InlinedArraysetImpl& other)
  : m_elementtype(other.m_elementtype),
    m_ndim(other.m_ndim),
    m_data(other.m_data)
{
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
}

iod::InlinedArraysetImpl::~InlinedArraysetImpl() { }

iod::InlinedArraysetImpl& iod::InlinedArraysetImpl::operator= 
(const InlinedArraysetImpl& other) {
  m_elementtype = other.m_elementtype;
  m_ndim = other.m_ndim;
  m_data = other.m_data;
  for (size_t i=0; i<m_ndim; ++i) m_shape[i] = other.m_shape[i];
  return *this;
}

const io::Array& iod::InlinedArraysetImpl::operator[] (size_t id) const {
  if (id >= m_data.size()) throw io::IndexError(id);
  return *(m_data[id].get());
}

io::Array& iod::InlinedArraysetImpl::operator[] (size_t id) {
  if (id >= m_data.size()) throw io::IndexError(id);
  return *(m_data[id].get());
}

boost::shared_ptr<const io::Array> iod::InlinedArraysetImpl::ptr
(size_t id) const {
  if (id >= m_data.size()) throw io::IndexError(id);
  return m_data[id];
}

boost::shared_ptr<io::Array> iod::InlinedArraysetImpl::ptr (size_t id) {
  if (id >= m_data.size()) throw io::IndexError(id);
  return m_data[id];
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
  m_data.push_back(boost::make_shared<io::Array>(array));
  return m_data.size()-1;
}

size_t iod::InlinedArraysetImpl::adopt (boost::shared_ptr<io::Array> array) {
  checkCompatibility(*array.get());
  updateTyping(*array.get());
  m_data.push_back(array);
  return m_data.size()-1;
}

void iod::InlinedArraysetImpl::remove(size_t id) {
  if (id >= m_data.size()) throw io::IndexError(id);
  std::vector<boost::shared_ptr<Torch::io::Array> >::iterator it=m_data.begin();
  it += id;
  m_data.erase(it);
  if (m_data.size() == 0) { //uninitialize
    m_elementtype = Torch::core::array::t_unknown;
    m_ndim = 0;
  }
}
