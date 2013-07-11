/**
 * @file ip/python/DCTFeatures.cc
 * @date Fri Apr 8 17:05:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the DCT features extractor to python
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

#include <bob/python/ndarray.h>
#include <bob/ip/DCTFeatures.h>

using namespace boost::python;

static const char* dctdoc = "Objects of this class, after configuration, can extract DCT features as described in the following reference.\n \"Polynomial Features for Robust Face Authentication\", C. Sanderson and K. Paliwal, in the proceedings of the IEEE International Conference on Image Processing 2002.\n This class also supports block normalization and DCT coefficient normalization.";

template <typename T> static object py_inner_2d_dct_apply(
  bob::ip::DCTFeatures& dct_features, bob::python::const_ndarray src,
  const bool output3d)
{
  if(output3d)
  {
    const blitz::TinyVector<int,3> shape = dct_features.get3DOutputShape(src.bz<T,2>());
    bob::python::ndarray dst(bob::core::array::t_float64, shape(0), shape(1), shape(2));
    blitz::Array<double,3> dst_ = dst.bz<double,3>();
    dct_features(src.bz<T,2>(), dst_);
    return dst.self();
  }
  else
  {
    const blitz::TinyVector<int,2> shape = dct_features.get2DOutputShape(src.bz<T,2>());
    bob::python::ndarray dst(bob::core::array::t_float64, shape(0), shape(1));
    blitz::Array<double,2> dst_ = dst.bz<double,2>();
    dct_features(src.bz<T,2>(), dst_);
    return dst.self();
  }
}

static object py_dct_apply(bob::ip::DCTFeatures& dct_features, 
  bob::python::const_ndarray src, const bool output3d=false)
{
  const bob::core::array::typeinfo& info = src.type();
  switch (info.dtype) {
    case bob::core::array::t_uint8: 
      return py_inner_2d_dct_apply<uint8_t>(dct_features, src, output3d);
    case bob::core::array::t_uint16:
      return py_inner_2d_dct_apply<uint16_t>(dct_features, src, output3d);
    case bob::core::array::t_float64: 
      return py_inner_2d_dct_apply<double>(dct_features, src, output3d);
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.DCTFeatures does not support input array of type '%s'.", info.str().c_str());
  }
}

template <typename T, int N> static object c_inner_dct_apply(
  bob::ip::DCTFeatures& dct_features, bob::python::const_ndarray src,
  bob::python::ndarray dst)
{
  blitz::Array<double,N> dst_ = dst.bz<double,N>();
  dct_features(src.bz<T,2>(), dst_);
  return dst.self();
}

static object c_dct_apply(bob::ip::DCTFeatures& dct_features, 
  bob::python::const_ndarray src, bob::python::ndarray dst)
{
  const bob::core::array::typeinfo& info = src.type();
  const bob::core::array::typeinfo& infod = dst.type();
  switch(infod.nd)
  {
    case 2:
      switch (info.dtype) {
        case bob::core::array::t_uint8: 
          return c_inner_dct_apply<uint8_t,2>(dct_features, src, dst);
        case bob::core::array::t_uint16:
          return c_inner_dct_apply<uint16_t,2>(dct_features, src, dst);
        case bob::core::array::t_float64: 
          return c_inner_dct_apply<double,2>(dct_features, src, dst);
        default: 
          PYTHON_ERROR(TypeError, "bob.ip.DCTFeatures does not support input array of type '%s'.", info.str().c_str());
      }
    case 3:
      switch (info.dtype) {
        case bob::core::array::t_uint8: 
          return c_inner_dct_apply<uint8_t,3>(dct_features, src, dst);
        case bob::core::array::t_uint16:
          return c_inner_dct_apply<uint16_t,3>(dct_features, src, dst);
        case bob::core::array::t_float64: 
          return c_inner_dct_apply<double,3>(dct_features, src, dst);
        default: 
          PYTHON_ERROR(TypeError, "bob.ip.DCTFeatures does not support input array of type '%s'.", info.str().c_str());
      }
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.DCTFeatures does not support output array with " SIZE_T_FMT " dimensions", infod.nd);
  }
}

static boost::python::tuple get_2d_output_shape(
  const bob::ip::DCTFeatures& dct_features, bob::python::const_ndarray input)
{
  // Computes output shape
  blitz::TinyVector<int,2> res;
  
  switch(input.type().dtype) 
  {
    case bob::core::array::t_uint8: 
      res = dct_features.get2DOutputShape(input.bz<uint8_t,2>());
      break;
    case bob::core::array::t_uint16:
      res = dct_features.get2DOutputShape(input.bz<uint16_t,2>());
      break;
    case bob::core::array::t_float64: 
      res = dct_features.get2DOutputShape(input.bz<double,2>());
      break;
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.DCTFeatures.get_2d_output_shape() does not support array of type '%s'.", input.type().str().c_str());
  }
   
  return boost::python::make_tuple(res[0], res[1]);
}

static boost::python::tuple get_3d_output_shape(
  const bob::ip::DCTFeatures& dct_features, bob::python::const_ndarray input)
{
  // Computes output shape
  blitz::TinyVector<int,3> res;
  
  switch(input.type().dtype) 
  {
    case bob::core::array::t_uint8: 
      res = dct_features.get3DOutputShape(input.bz<uint8_t,2>());
      break;
    case bob::core::array::t_uint16:
      res = dct_features.get3DOutputShape(input.bz<uint16_t,2>());
      break;
    case bob::core::array::t_float64: 
      res = dct_features.get3DOutputShape(input.bz<double,2>());
      break;
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.DCTFeatures.get_3d_output_shape() does not support array of type '%s'.", input.type().str().c_str());
  }
   
  return boost::python::make_tuple(res[0], res[1], res[2]);
}


void bind_ip_dctfeatures() 
{
  class_<bob::ip::DCTFeatures, boost::shared_ptr<bob::ip::DCTFeatures> >("DCTFeatures", dctdoc, init<const size_t, const size_t, const size_t, const size_t, const size_t, optional<const bool, const bool, const bool> >((arg("self"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w"), arg("n_dct_coefs"), arg("norm_block")=false, arg("norm_dct")=false, arg("square_pattern")=false), "Constructs a new DCT features extractor.")) 
    .def(init<bob::ip::DCTFeatures&>((arg("self"), arg("other"))))
    .def(self == self)
    .def(self != self)
    .add_property("block_h", &bob::ip::DCTFeatures::getBlockH, &bob::ip::DCTFeatures::setBlockH, "The height of each block for the block decomposition")
    .add_property("block_w", &bob::ip::DCTFeatures::getBlockW, &bob::ip::DCTFeatures::setBlockW, "The width of each block for the block decomposition")
    .add_property("overlap_h", &bob::ip::DCTFeatures::getOverlapH, &bob::ip::DCTFeatures::setOverlapH, "The overlap of the blocks along the y-axis")
    .add_property("overlap_w", &bob::ip::DCTFeatures::getOverlapW, &bob::ip::DCTFeatures::setOverlapW, "The overlap of the blocks along the x-axis")
    .add_property("n_dct_coefs", &bob::ip::DCTFeatures::getNDctCoefs, &bob::ip::DCTFeatures::setNDctCoefs, "The number of DCT coefficients. The real number of DCT coefficient returned by the extractor is n_dct_coefs-1 when the block normalization is enabled (as the first coefficient is always 0 in this case).")
    .add_property("norm_block", &bob::ip::DCTFeatures::getNormalizeBlock, &bob::ip::DCTFeatures::setNormalizeBlock, "Normalize each block to zero mean and unit variance before extracting DCT coefficients. In this case, the first coefficient will always be zero and hence will not be returned.")
    .add_property("norm_dct", &bob::ip::DCTFeatures::getNormalizeDct, &bob::ip::DCTFeatures::setNormalizeDct, "Normalize DCT coefficients to zero mean and unit variance after the DCT extraction")
    .add_property("square_pattern", &bob::ip::DCTFeatures::getSquarePattern, &bob::ip::DCTFeatures::setSquarePattern, "Tells whether a zigzag pattern or a square pattern is used for the DCT extraction. For a square pattern, the number of DCT coefficients must be a square integer.")
    .add_property("norm_epsilon", &bob::ip::DCTFeatures::getNormEpsilon, &bob::ip::DCTFeatures::setNormEpsilon, "The epsilon value to avoid division-by-zero when performing block or DCT coefficient normalization")
    .def("get_2d_output_shape", &get_2d_output_shape, "Returns the expected shape of the 2D destination array when extracting DCT features.")
    .def("get_3d_output_shape", &get_3d_output_shape, "Returns the expected shape of the 3D destination array when extracting DCT features.")
    .def("__call__", &py_dct_apply, (arg("self"), arg("src"), arg("output3d")=false), "Extracts DCT features from either uint8, uint16 or double arrays. The input numpy.array a 2D array/grayscale image. This method returns a 2D numpy.array with these DCT features.")
    .def("__call__", &c_dct_apply, (arg("self"), arg("src"), arg("dst")), "Extracts DCT features from either uint8, uint16 or double arrays. The input numpy.array a 2D array/grayscale image. The destination array should be a 2D or 3D numpy array of type numpy.float64 and allocated with the correct dimensions.")
  ;
}
