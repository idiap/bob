/**
 * @file core/cxx/blitz_array.cc
 * @date Tue Nov 8 15:34:31 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Implementation of non-templated methods of the blitz
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "bob/core/blitz_array.h"
#include "bob/core/logging.h"

namespace ca = bob::core::array;

ca::blitz_array::blitz_array(boost::shared_ptr<blitz_array> other) {
  set(other);
}

ca::blitz_array::blitz_array(const blitz_array& other) {
  set(other);
}

ca::blitz_array::blitz_array(boost::shared_ptr<interface> other) {
  set(other);
}

ca::blitz_array::blitz_array(const interface& other) {
  set(other);
}

ca::blitz_array::blitz_array(const typeinfo& info) {
  set(info);
}

ca::blitz_array::blitz_array(void* data, const typeinfo& info):
  m_type(info),
  m_ptr(data),
  m_is_blitz(false) {
}

ca::blitz_array::~blitz_array() {
}

void ca::blitz_array::set(boost::shared_ptr<blitz_array> other) {
  m_type = other->m_type;
  m_ptr = other->m_ptr;
  m_is_blitz = other->m_is_blitz;
  m_data = other->m_data;
}

void ca::blitz_array::set(const interface& other) {
  TDEBUG1("[non-optimal] buffer data copy requested: " << other_type.str());
  set(other.type());
  memcpy(m_ptr, other.ptr(), m_type.buffer_size());
}

void ca::blitz_array::set(boost::shared_ptr<interface> other) {
  m_type = other->type();
  m_ptr = other->ptr();
  m_is_blitz = false;
  m_data = other;
}

template <typename T>
static boost::shared_ptr<void> make_array(size_t nd, const size_t* shape,
    void*& ptr) {
  switch(nd) {
    case 1:
      {
        blitz::TinyVector<int,1> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval = 
          boost::make_shared<blitz::Array<T,1> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,1> >(retval)->data()); 
        return retval;
      }
    case 2:
      {
        blitz::TinyVector<int,2> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval = 
          boost::make_shared<blitz::Array<T,2> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,2> >(retval)->data()); 
        return retval;
      }
    case 3:
      {
        blitz::TinyVector<int,3> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval = 
          boost::make_shared<blitz::Array<T,3> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,3> >(retval)->data()); 
        return retval;
      }
    case 4:
      {
        blitz::TinyVector<int,4> tv_shape;
        for (size_t k=0; k<nd; ++k) tv_shape[k] = shape[k];
        boost::shared_ptr<void> retval = 
          boost::make_shared<blitz::Array<T,4> >(tv_shape);
        ptr = reinterpret_cast<void*>(boost::static_pointer_cast<blitz::Array<T,4> >(retval)->data()); 
        return retval;
      }
    default:
      break;
  }
  throw std::runtime_error("unsupported number of dimensions -- debug me");
}

void ca::blitz_array::set (const ca::typeinfo& req) {
  if (m_type.is_compatible(req)) return; ///< double-check requirement first!

  //ok, have to go through reallocation
  TDEBUG1("[non-optimal] buffer re-allocation requested from " << m_type.str() << " to " << req.str());
  m_type = req;
  m_is_blitz = true;
  switch (m_type.dtype) {
    case ca::t_bool:
      m_data = make_array<bool>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_int8: 
      m_data = make_array<int8_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_int16: 
      m_data = make_array<int16_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_int32: 
      m_data = make_array<int32_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_int64: 
      m_data = make_array<int64_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_uint8: 
      m_data = make_array<uint8_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_uint16: 
      m_data = make_array<uint16_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_uint32: 
      m_data = make_array<uint32_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_uint64: 
      m_data = make_array<uint64_t>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_float32: 
      m_data = make_array<float>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_float64: 
      m_data = make_array<double>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_float128: 
      m_data = make_array<long double>(req.nd, req.shape, m_ptr);
      return;
    case ca::t_complex64: 
      m_data = make_array<std::complex<float> >(req.nd, req.shape, m_ptr);
      return;
    case ca::t_complex128: 
      m_data = make_array<std::complex<double> >(req.nd, req.shape, m_ptr);
      return;
    case ca::t_complex256: 
      m_data = make_array<std::complex<long double> >(req.nd, req.shape, m_ptr);
      return;
    default:
      break;
  }

  //if we get to this point, there is nothing much we can do...
  throw std::runtime_error("invalid data type on blitz array reset -- debug me");
}
