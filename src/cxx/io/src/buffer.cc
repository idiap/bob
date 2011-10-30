/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Thu  6 Oct 12:26:01 2011
 *
 * @brief Some buffer stuff
 */

#include "io/buffer.h"

namespace io = Torch::io;

Torch::core::array::ElementType dtype; ///< data type
size_t nd; ///< number of dimensions
size_t shape[TORCH_MAX_DIM]; ///< length along each dimension
size_t stride[TORCH_MAX_DIM]; ///< strides along each dimension

io::typeinfo::typeinfo():
  dtype(Torch::core::array::t_unknown),
  nd(0)
{
}

io::typeinfo::typeinfo(const io::typeinfo& other): 
  dtype(other.dtype)
{
  set_shape(other.nd, other.shape);
}

io::typeinfo& io::typeinfo::operator= (const io::typeinfo& other) {
  dtype = other.dtype;
  set_shape(other.nd, other.shape);
  return *this;
}

void io::typeinfo::reset() {
  dtype = Torch::core::array::t_unknown;
  nd = 0;
}

bool io::typeinfo::is_valid() const {
  return (dtype != Torch::core::array::t_unknown) && (nd > 0) && (nd <= TORCH_MAX_DIM);
}

void io::typeinfo::update_strides() {
  switch (nd) {
    case 0:
      return;
    case 1:
      stride[0] = 1;
      return;
    case 2:
      stride[1] = 1;
      stride[0] = shape[1];
      return;
    case 3:
      stride[2] = 1;
      stride[1] = shape[2];
      stride[0] = shape[1]*shape[2];
      return;
    case 4:
      stride[3] = 1;
      stride[2] = shape[3];
      stride[1] = shape[2]*shape[3];
      stride[0] = shape[1]*shape[2]*shape[3];
      return;
    default:
      break;
  }
  throw std::invalid_argument("unsupported number of dimensions");
}

size_t io::typeinfo::size() const {
  size_t retval = 1;
  for (size_t k=0; k<nd; ++k) retval *= shape[k];
  return retval;
}

size_t io::typeinfo::buffer_size() const {
  return size()*Torch::core::array::getElementSize(dtype);
}

static bool same_shape(size_t nd, const size_t* s1, const size_t* s2) {
  for (size_t k=0; k<nd; ++k) if (s1[k] != s2[k]) return false;
  return true;
}

bool io::typeinfo::is_compatible(const io::typeinfo& other) const {
  return (dtype == other.dtype) && (nd == other.nd) && same_shape(nd, shape, other.shape);
}
