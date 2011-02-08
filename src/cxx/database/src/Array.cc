/**
 * @file database/src/Array.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implementation of the Array class.
 */

#include "database/Array.h"
//#include "database/Arrayset.h"

namespace db = Torch::database;

db::Array::Array(const std::string& filename, const std::string& codec)
  : m_external(new db::detail::ExternalArrayImpl(filename, codec)),
    m_id(0)
{
}

db::Array::Array(const Array& other)
  : //m_parent_arrayset(other.m_parent_arrayset),
    m_inlined(other.m_inlined),
    m_external(other.m_external),
    m_id(0)
{
  //if (m_parent_arrayset) m_parent_arrayset->addArray(*this);
}

db::Array::~Array() {
  //m_parent_arrayset->removeArray(*this);
}

db::Array& db::Array::operator= (const db::Array& other) {
  //m_parent_arrayset = other.m_parent_arrayset;
  m_inlined = other.m_inlined;
  m_external = other.m_external;
  m_id = 0;
  //if (m_parent_arrayset) m_parent_arrayset->addArray(*this);
  return *this;
}

size_t db::Array::getNDim() const {
  if (m_inlined) return m_inlined->getNDim(); 
  Torch::core::array::ElementType eltype;
  size_t ndim;
  m_external->getSpecification(eltype, ndim);
  return ndim;
}

Torch::core::array::ElementType db::Array::getElementType() const {
  if (m_inlined) return m_inlined->getElementType(); 
  Torch::core::array::ElementType eltype;
  size_t ndim;
  m_external->getSpecification(eltype, ndim);
  return eltype;
}

const size_t* db::Array::getShape() const {
  if (m_inlined) return m_inlined->getShape(); 
  Torch::core::array::ElementType eltype;
  size_t ndim;
  m_external->getSpecification(eltype, ndim, m_tmp_shape);
  return m_tmp_shape;
}

void db::Array::save(const std::string& filename, const std::string& codecname) 
{
  if (m_inlined) {
    m_external.reset(new db::detail::ExternalArrayImpl(filename, codecname));
    m_external->save(*m_inlined);
    m_inlined.reset();
    return;
  }
  m_external->move(filename, codecname); 
}

/**
void db::Array::setParent (boost::shared_ptr<Arrayset> parent, size_t id) {
  //first we double check that the parent has equivalent typing information
  if (parent->getNDim() != getNDim()) throw DimensionError();
  if (parent->getElementType() != getElementType()) throw TypeError();
  //if so, we exchange, first de-registering the array from the parent
  m_parent_arrayset->removeArray(*this);
  m_parent_arrayset = parent;
  m_id = id;
  m_parent_arrayset->addArray(*this);
}
**/
        
const std::string& db::Array::getFilename() const {
  if (m_external) return m_external->getFilename();
  static std::string empty_string;
  return empty_string;
}

boost::shared_ptr<const db::ArrayCodec> db::Array::getCodec() const {
  if (m_external) return m_external->getCodec();
  return boost::shared_ptr<ArrayCodec>(); 
}
