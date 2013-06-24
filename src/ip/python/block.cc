/**
 * @file ip/python/block.cc
 * @date Sun Jun 26 18:59:21 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds block decomposition into python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include "bob/python/ndarray.h"
#include "bob/ip/block.h"

using namespace boost::python;

static const char* BLOCK2D_DOC = 
  "Performs a block decomposition of a 2D array/image. The output 3D or 4D \
  destination array should be allocated and of the correct size.";
static const char* BLOCK2D_P_DOC = 
  "Performs a block decomposition of a 2D array/image. This will allocate \
  and return a 4D array of blocks indexed along the y- and x-axes \
  (block_index_y, block_index_x, y, x).";
static const char* GETBLOCK3DOUTPUTSHAPE_DOC = "Returns the shape of the \
  output 2D blitz array/image, when calling bob.ip.block() which performs a \
  block decomposition of a 2D array/image, and saving the results in a 3D \
  array (block_index, y, x).";
static const char* GETBLOCK4DOUTPUTSHAPE_DOC = "Returns the shape of the \
  output 2D blitz array/image, when calling bob.ip.block() which performs a \
  block decomposition of a 2D array/image, and saving the results in a 4D \
  array (block_index_y, block_index_x, y, x).";

template <typename T> 
static void inner_block_3d(bob::python::const_ndarray input,
  bob::python::ndarray output, const size_t a, const size_t b, 
  const size_t c, const size_t d) 
{
  blitz::Array<T,3> output_ = output.bz<T,3>();
  bob::ip::block<T>(input.bz<T,2>(), output_, a, b, c, d);
}

template <typename T> 
static void inner_block_4d(bob::python::const_ndarray input,
  bob::python::ndarray output, const size_t a, const size_t b, 
  const size_t c, const size_t d) 
{
  blitz::Array<T,4> output_ = output.bz<T,4>();
  bob::ip::block<T>(input.bz<T,2>(), output_, a, b, c, d);
}

static void block(bob::python::const_ndarray input, 
  bob::python::ndarray output, const size_t a, const size_t b, 
  const size_t c, const size_t d)
{
  const bob::core::array::typeinfo& infoOut = output.type();
  const bob::core::array::typeinfo& info = input.type();
  switch(infoOut.nd) {
    case 3:
      switch (info.dtype) {
        case bob::core::array::t_uint8: 
          return inner_block_3d<uint8_t>(input, output, a,b,c,d);
        case bob::core::array::t_uint16:
          return inner_block_3d<uint16_t>(input, output, a,b,c,d);
        case bob::core::array::t_float64: 
          return inner_block_3d<double>(input, output, a,b,c,d);
        default: 
          PYTHON_ERROR(TypeError, 
            "bob.ip.block() does not support array with type '%s'.", 
            info.str().c_str());
      }
    case 4:
      switch (info.dtype) {
        case bob::core::array::t_uint8: 
          return inner_block_4d<uint8_t>(input, output, a,b,c,d);
        case bob::core::array::t_uint16:
          return inner_block_4d<uint16_t>(input, output, a,b,c,d);
        case bob::core::array::t_float64: 
          return inner_block_4d<double>(input, output, a,b,c,d);
        default: 
          PYTHON_ERROR(TypeError, 
            "bob.ip.block() does not support array with type '%s'.", 
            info.str().c_str());
      }     
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.block() operator does not support output array with number \
         of dimensions " SIZE_T_FMT ".", info.nd);
  }
}

static object block_p(bob::python::const_ndarray input, 
  const size_t a, const size_t b, const size_t c, const size_t d)
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8:
      {
        const blitz::TinyVector<int,4> shape = 
          bob::ip::getBlock4DOutputShape<uint8_t>(input.bz<uint8_t,2>(), 
            a, b, c, d);
        bob::python::ndarray output(bob::core::array::t_uint8, shape(0), 
          shape(1), shape(2), shape(3));
        inner_block_4d<uint8_t>(input, output, a,b,c,d);
        return output.self();
      }
    case bob::core::array::t_uint16:
      {
        const blitz::TinyVector<int,4> shape = 
          bob::ip::getBlock4DOutputShape<uint16_t>(input.bz<uint16_t,2>(), 
            a, b, c, d);
        bob::python::ndarray output(bob::core::array::t_uint16, shape(0), 
          shape(1), shape(2), shape(3));
        inner_block_4d<uint16_t>(input, output, a,b,c,d);
        return output.self();
      }
    case bob::core::array::t_float64: 
      {
        const blitz::TinyVector<int,4> shape = 
          bob::ip::getBlock4DOutputShape<double>(input.bz<double,2>(), 
            a, b, c, d);
        bob::python::ndarray output(bob::core::array::t_float64, shape(0), 
          shape(1), shape(2), shape(3));
        inner_block_4d<double>(input, output, a,b,c,d);
        return output.self();
      }
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.block() does not support array with type '%s'.", 
        info.str().c_str());
  }
}


template <typename T> 
static object inner_get_block_3d_output_shape(
  bob::python::const_ndarray input, const size_t a, const size_t b, 
  const size_t c, const size_t d) 
{
  return object(bob::ip::getBlock3DOutputShape<T>(input.bz<T,2>(), a, b, c, d));
}

template <typename T> 
static object inner_get_block_4d_output_shape(
  bob::python::const_ndarray input, const size_t a, const size_t b, 
  const size_t c, const size_t d) 
{
  return object(bob::ip::getBlock4DOutputShape<T>(input.bz<T,2>(), a, b, c, d));
}

static object get_block_3d_output_shape(bob::python::const_ndarray input,
  const size_t a, const size_t b, const size_t c, const size_t d) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_get_block_3d_output_shape<uint8_t>(input, a,b,c,d);
    case bob::core::array::t_uint16:
      return inner_get_block_3d_output_shape<uint16_t>(input, a,b,c,d);
    case bob::core::array::t_float64: 
      return inner_get_block_3d_output_shape<double>(input, a,b,c,d);
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.get_block_3d_output_shape() does not support array with \
         type '%s'.", info.str().c_str());
  }
}

static object get_block_4d_output_shape(bob::python::const_ndarray input,
  const size_t a, const size_t b, const size_t c, const size_t d) 
{
  const bob::core::array::typeinfo& info = input.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return inner_get_block_4d_output_shape<uint8_t>(input, a,b,c,d);
    case bob::core::array::t_uint16:
      return inner_get_block_4d_output_shape<uint16_t>(input, a,b,c,d);
    case bob::core::array::t_float64: 
      return inner_get_block_4d_output_shape<double>(input, a,b,c,d);
    default: 
      PYTHON_ERROR(TypeError, 
        "bob.ip.get_block_4d_output_shape() does not support array with \
         type '%s'", info.str().c_str());
  }
}


void bind_ip_block()
{
  def("block", &block, (arg("src"), arg("dst"), arg("block_h"), 
    arg("block_w"), arg("overlap_h"), arg("overlap_w")), BLOCK2D_DOC);
  def("block", &block_p, (arg("src"), arg("block_h"), arg("block_w"), 
    arg("overlap_h"), arg("overlap_w")), BLOCK2D_P_DOC);
  def("get_block_3d_output_shape", &get_block_3d_output_shape, 
    (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), 
    arg("overlap_w")), GETBLOCK3DOUTPUTSHAPE_DOC);
  def("get_block_4d_output_shape", &get_block_4d_output_shape, 
    (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), 
    arg("overlap_w")), GETBLOCK4DOUTPUTSHAPE_DOC);
}
