/**
 * @file python/ip/src/block.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds block decomposition into python
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

#include "core/python/ndarray.h"
#include "ip/block.h"

using namespace boost::python;
namespace ip = bob::ip;
namespace tp = bob::python;
namespace ca = bob::core::array;

using namespace boost::python;

static const char* BLOCK2D_DOC = "Performs a block decomposition of a 2D array/image.";
static const char* GETBLOCK3DOUTPUTSHAPE_DOC = "Returns the shape of the output 2D blitz\
  array/image, when calling block which performs a block decomposition of a 2D array/image,\
  and saving the results in a 3D array.";
static const char* GETBLOCK4DOUTPUTSHAPE_DOC = "Returns the shape of the output 2D blitz\
  array/image, when calling block which performs a block decomposition of a 2D array/image,\
  and saving the results in a 4D array.";

template <typename T> static void inner_block_3d(tp::const_ndarray input,
    tp::ndarray output, int a, int b, int c, int d) {
  blitz::Array<T,3> output_ = output.bz<T,3>();
  ip::block<T>(input.bz<T,2>(), output_, a, b, c, d);
}

template <typename T> static void inner_block_4d(tp::const_ndarray input,
    tp::ndarray output, int a, int b, int c, int d) {
  blitz::Array<T,4> output_ = output.bz<T,4>();
  ip::block<T>(input.bz<T,2>(), output_, a, b, c, d);
}

static void block (tp::const_ndarray input, tp::ndarray output,
    int a, int b, int c, int d) {
  const ca::typeinfo& infoOut = output.type();
  const ca::typeinfo& info = input.type();
  switch(infoOut.nd) {
    case 3:
      switch (info.dtype) {
        case ca::t_uint8: 
          return inner_block_3d<uint8_t>(input, output, a,b,c,d);
        case ca::t_uint16:
          return inner_block_3d<uint16_t>(input, output, a,b,c,d);
        case ca::t_float64: 
          return inner_block_3d<double>(input, output, a,b,c,d);
        default: PYTHON_ERROR(TypeError, "block operator does not support array with type '%s'", info.str().c_str());
      }
    case 4:
      switch (info.dtype) {
        case ca::t_uint8: 
          return inner_block_4d<uint8_t>(input, output, a,b,c,d);
        case ca::t_uint16:
          return inner_block_4d<uint16_t>(input, output, a,b,c,d);
        case ca::t_float64: 
          return inner_block_4d<double>(input, output, a,b,c,d);
        default: PYTHON_ERROR(TypeError, "block operator does not support array with type '%s'", info.str().c_str());
      }     
    default: PYTHON_ERROR(TypeError, "block operator does not support output array with number of dimensions '%d'", info.nd);
  }
}


template <typename T> static object inner_get_block_3d_output_shape 
(tp::const_ndarray input, int a, int b, int c, int d) {
  return object(ip::getBlock3DOutputShape<T>(input.bz<T,2>(), a, b, c, d));
}

template <typename T> static object inner_get_block_4d_output_shape 
(tp::const_ndarray input, int a, int b, int c, int d) {
  return object(ip::getBlock4DOutputShape<T>(input.bz<T,2>(), a, b, c, d));
}

static object get_block_3d_output_shape (tp::const_ndarray input,
    int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_get_block_3d_output_shape<uint8_t>(input, a,b,c,d);
    case ca::t_uint16:
      return inner_get_block_3d_output_shape<uint16_t>(input, a,b,c,d);
    case ca::t_float64: 
      return inner_get_block_3d_output_shape<double>(input, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "operation does not support array with type '%s'", info.str().c_str());
  }
}

static object get_block_4d_output_shape (tp::const_ndarray input,
    int a, int b, int c, int d) {
  const ca::typeinfo& info = input.type();
  switch (info.dtype) {
    case ca::t_uint8: 
      return inner_get_block_4d_output_shape<uint8_t>(input, a,b,c,d);
    case ca::t_uint16:
      return inner_get_block_4d_output_shape<uint16_t>(input, a,b,c,d);
    case ca::t_float64: 
      return inner_get_block_4d_output_shape<double>(input, a,b,c,d);
    default: PYTHON_ERROR(TypeError, "operation does not support array with type '%s'", info.str().c_str());
  }
}


void bind_ip_block()
{
  def("block", &block, (arg("src"), arg("dst"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), BLOCK2D_DOC);
  def("get_block_3d_output_shape", &get_block_3d_output_shape, (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), GETBLOCK3DOUTPUTSHAPE_DOC);
  def("get_block_4d_output_shape", &get_block_4d_output_shape, (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), GETBLOCK4DOUTPUTSHAPE_DOC);
}
