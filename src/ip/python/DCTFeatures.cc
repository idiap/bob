/**
 * @file ip/python/DCTFeatures.cc
 * @date Fri Apr 8 17:05:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the DCT features extractor to python
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

#include "bob/ip/DCTFeatures.h"
#include "bob/core/python/ndarray.h"

using namespace boost::python;

static const char* dctdoc = "Objects of this class, after configuration, extract DCT features as described in the paper titled \"Polynomial Features for Robust Face Authentication\", published in 2002.";

template <typename T> static object inner_dct_apply(
  bob::ip::DCTFeatures& dct_features, bob::python::const_ndarray src,
  const bool output_array)
{
  if(output_array)
  {
    bob::python::ndarray dst(bob::core::array::t_float64, 
      dct_features.getNBlocks(src.bz<T,2>()), dct_features.getNDctCoefs());
    blitz::Array<double,2> dst_ = dst.bz<double,2>();
    dct_features(src.bz<T,2>(), dst_);
    return dst.self();
  }
  else
  {
    std::vector<blitz::Array<double,1> > dst;
    dct_features(src.bz<T,2>(), dst);
    list t;
    for(size_t i=0; i<dst.size(); ++i) t.append(dst[i]); //bz array copying!
    return t;
  }
}

template <typename T> static object inner_dct_apply3
(bob::ip::DCTFeatures& dct_features, bob::python::const_ndarray src) 
{
  blitz::Array<double,2> dst;
  dct_features(src.bz<T,3>(), dst);
  return object(dst);
}

static object dct_apply(bob::ip::DCTFeatures& dct_features, 
  bob::python::const_ndarray src, const bool output_array=false)
{
  const bob::core::array::typeinfo& info = src.type();
  switch(info.nd)
  {
    case 2:
      switch (info.dtype) {
        case bob::core::array::t_uint8: 
          return inner_dct_apply<uint8_t>(dct_features, src, output_array);
        case bob::core::array::t_uint16:
          return inner_dct_apply<uint16_t>(dct_features, src, output_array);
        case bob::core::array::t_float64: 
          return inner_dct_apply<double>(dct_features, src, output_array);
        default: 
          PYTHON_ERROR(TypeError, "DCT feature extraction does not support input array of type '%s'.", info.str().c_str());
      }
    case 3:
      switch (info.dtype) {
        case bob::core::array::t_uint8: 
          return inner_dct_apply3<uint8_t>(dct_features, src);
        case bob::core::array::t_uint16:
          return inner_dct_apply3<uint16_t>(dct_features, src);
        case bob::core::array::t_float64: 
          return inner_dct_apply3<double>(dct_features, src);
        default: PYTHON_ERROR(TypeError, "DCT feature extraction does not input array of type '%s'", info.str().c_str());
      }
    default:
      PYTHON_ERROR(TypeError, "DCT feature extraction does not support input array with " SIZE_T_FMT " dimensions", info.nd);
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(dct_apply_overloads, dct_apply, 2, 3)

void bind_ip_dctfeatures() 
{
  class_<bob::ip::DCTFeatures, boost::shared_ptr<bob::ip::DCTFeatures> >("DCTFeatures", dctdoc, init<const size_t, const size_t, const size_t, const size_t, const size_t, optional<const bool, const bool> >((arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w"), arg("n_dct_coefs"), arg("norm_block")=false, arg("norm_dct")=false), "Constructs a new DCT features extractor.")) 
    .def(init<bob::ip::DCTFeatures&>(args("other")))
    .def(self == self)
    .def(self != self)
    .add_property("block_h", &bob::ip::DCTFeatures::getBlockH, &bob::ip::DCTFeatures::setBlockH, "The height of each block for the block decomposition")
    .add_property("block_w", &bob::ip::DCTFeatures::getBlockW, &bob::ip::DCTFeatures::setBlockW, "The width of each block for the block decomposition")
    .add_property("overlap_h", &bob::ip::DCTFeatures::getOverlapH, &bob::ip::DCTFeatures::setOverlapH, "The overlap of the blocks along the y-axis")
    .add_property("overlap_w", &bob::ip::DCTFeatures::getOverlapW, &bob::ip::DCTFeatures::setOverlapW, "The overlap of the blocks along the x-axis")
    .add_property("n_dct_coefs", &bob::ip::DCTFeatures::getNDctCoefs, &bob::ip::DCTFeatures::setNDctCoefs, "The number of DCT coefficients")
    .add_property("norm_block", &bob::ip::DCTFeatures::getNormalizeBlock, &bob::ip::DCTFeatures::setNormalizeBlock, "Normalize each block to zero mean and unit variance before extracting DCT coefficients")
    .add_property("norm_dct", &bob::ip::DCTFeatures::getNormalizeDct, &bob::ip::DCTFeatures::setNormalizeDct, "Normalize DCT coefficients to zero mean and unit variance after the DCT extraction")
    .def("__call__", &dct_apply, dct_apply_overloads((arg("self"), arg("input"), arg("output_array")=false), "Extracts DCT features from either uint8, uint16 or double arrays. The input numpy.array can be a 2D array, which will be interpreted as a grayscale image, OR a 3D array, which will be interpreted as a set of 2D blocks. This method returns a list of DCT features in a 1D numpy.array, or a 2D numpy.array with these DCT features  if the output_array argument is enabled AND the input is a 2D image array."))
    ;
}
