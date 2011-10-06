/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  5 Oct 12:34:11 2011
 *
 * @brief Implementation of non-templated methods of the carray
 */

#include "io/carray.h"

namespace io = Torch::io;

io::carray::carray(boost::shared_ptr<io::carray> other) {
  set(other);
}

io::carray::carray(const io::carray& other) {
  set(other);
}

io::carray::carray(boost::shared_ptr<io::buffer> other) {
  set(other);
}

io::carray::carray(const io::buffer& other) {
  set(other);
}

io::carray::carray(const typeinfo& info) {
  set(info);
}

io::carray::~carray() {
}

void io::carray::set(boost::shared_ptr<io::carray> other) {
  m_type = other->m_type;
  m_ptr = other->m_ptr;
  m_is_blitz = other->m_is_blitz;
  m_data = other->m_data;
}

void io::carray::set(const io::buffer& other) {
  set(other.type());
  memcpy(m_ptr, other.ptr(), m_type.buffer_size());
}

void io::carray::set(boost::shared_ptr<io::buffer> other) {
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

void io::carray::set (const io::typeinfo& req) {
  m_type = req;
  m_is_blitz = true;
  switch (m_type.dtype) {
    case Torch::core::array::t_bool:
      m_data = make_array<bool>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_int8: 
      m_data = make_array<int8_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_int16: 
      m_data = make_array<int16_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_int32: 
      m_data = make_array<int32_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_int64: 
      m_data = make_array<int64_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_uint8: 
      m_data = make_array<uint8_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_uint16: 
      m_data = make_array<uint16_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_uint32: 
      m_data = make_array<uint32_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_uint64: 
      m_data = make_array<uint64_t>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_float32: 
      m_data = make_array<float>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_float64: 
      m_data = make_array<double>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_float128: 
      m_data = make_array<long double>(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_complex64: 
      m_data = make_array<std::complex<float> >(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_complex128: 
      m_data = make_array<std::complex<double> >(req.nd, req.shape, m_ptr);
      return;
    case Torch::core::array::t_complex256: 
      m_data = make_array<std::complex<long double> >(req.nd, req.shape, m_ptr);
      return;
    default:
      break;
  }

  //if we get to this point, there is nothing much we can do...
  throw std::runtime_error("invalid data type on blitz buffer reset -- debug me");
}
