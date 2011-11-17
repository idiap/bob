/**
 * @file src/python/ip/src/block.cc 
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds block decomposition into python 
 */

#include "core/python/ndarray.h"
#include "ip/block.h"

using namespace boost::python;
namespace ip = Torch::ip;
namespace tp = Torch::python;
namespace ca = Torch::core::array;

using namespace boost::python;

static const char* BLOCK2D_DOC = "Perform a block decomposition of a 2D blitz array/image.";
static const char* GETBLOCKSHAPE2D_DOC = "Return the shape of the output 2D blitz array/image, when calling block which performs a block decomposition of a 2D blitz array/image.";
static const char* GETNBLOCKS2D_DOC = "Return the number of blocks of the output 2D blitz array/image, when calling block which performs a block decomposition of a 2D blitz array/image.";

template <typename T> static void inner_block (tp::const_ndarray input,
    tp::ndarray output, int a, int b, int c, int d) {
  blitz::Array<T,3> output_ = output.bz<T,3>();
  ip::block<T>(input.bz<T,2>(), output_, a, b, c, d);
}

static void block (tp::const_ndarray input, tp::ndarray output,
    int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_block<uint8_t>(input, output, a,b,c,d);
    case ca::t_uint16:
      return inner_block<uint16_t>(input, output, a,b,c,d);
    case ca::t_float64: 
      return inner_block<double>(input, output, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "block operator does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static object inner_get_block_shape 
(tp::const_ndarray input, int a, int b, int c, int d) {
  return object(ip::getBlockShape<T>(input.bz<T,2>(), a, b, c, d));
}

static object get_block_shape (tp::const_ndarray input,
    int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_get_block_shape<uint8_t>(input, a,b,c,d);
    case ca::t_uint16:
      return inner_get_block_shape<uint16_t>(input, a,b,c,d);
    case ca::t_float64: 
      return inner_get_block_shape<double>(input, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "operation does not support array with type '%s'", info.str().c_str());
  }
}

template <typename T> static object inner_get_n_blocks
(tp::const_ndarray input, int a, int b, int c, int d) {
  return object(ip::getNBlocks<T>(input.bz<T,2>(), a, b, c, d));
}

static object get_n_blocks (tp::const_ndarray input, int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_get_n_blocks<uint8_t>(input, a,b,c,d);
    case ca::t_uint16:
      return inner_get_n_blocks<uint16_t>(input, a,b,c,d);
    case ca::t_float64: 
      return inner_get_n_blocks<double>(input, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "operation does not support array with type '%s'", info.str().c_str());
  }
}


void bind_ip_block()
{
  def("block", &block, (arg("src"), arg("dst"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), BLOCK2D_DOC);
  def("getBlockShape", &get_block_shape, (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), GETBLOCKSHAPE2D_DOC);
  def("getNBlocks", &get_n_blocks, (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), GETNBLOCKS2D_DOC);
}
