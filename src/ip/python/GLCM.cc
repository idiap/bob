/**
 * @file ip/python/GLCM.cc
 * @date Wed Jan 23 19:49:22 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 *
 * @brief Binds the GLCM class to python
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

#include "bob/ip/GLCM.h"

#include "bob/core/python/ndarray.h"
#include <boost/make_shared.hpp>

using namespace boost::python;

static const char* glcm_doc = "Objects of this class, after configuration, can compute Grey-Level Co-occurence Matrix of an image.\n This class allows to extract Grey-Level Co-occurence Matrix (GLCM). A thorough tutorial about GLCM and the textural (so-called Haralick) properties that can be derived from it, can be found at: http://www.fp.ucalgary.ca/mhallbey/tutorial.htm \n List of references: \n [1] R. M. Haralick, K. Shanmugam, I. Dinstein; \"Textural Features for Image calssification\", in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621. [2] http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html";


static boost::shared_ptr<bob::ip::GLCM> py_constructor_glcm() 
{  
  return boost::make_shared<bob::ip::GLCM>();
}
  
static void call_set_offset(bob::ip::GLCM& op, bob::python::const_ndarray input)
{
  blitz::Array<int32_t, 2> input_ = input.bz<int32_t,2>(); 
  op.setOffset(input_);
} 


static object call_get_offset(const bob::ip::GLCM& op)
{
  blitz::Array<int32_t,2> output=op.getOffset();
  return object(output);
}  

      
template <typename T>
static const blitz::TinyVector<int,3> inner_call_getshape(const bob::ip::GLCM& op, bob::python::const_ndarray input)
{
  return op.getGLCMShape(input.bz<T,2>());
} 

static object call_get_shape(const bob::ip::GLCM& op, bob::python::const_ndarray input) 
{
  blitz::TinyVector<int,3> output;
  switch(input.type().dtype) {
    case bob::core::array::t_uint8: 
      output = inner_call_getshape<uint8_t>(op, input);
      break;
    case bob::core::array::t_uint16:
      output = inner_call_getshape<uint16_t>(op, input);
      break;
    default:
      PYTHON_ERROR(TypeError, "GLCM operator cannot process image of type '%s'", input.type().str().c_str());
  }
  return object(output);
}


template <typename T> 
static void inner_call_glcm(const bob::ip::GLCM& op, bob::python::const_ndarray input, bob::python::ndarray output) 
{
  blitz::Array<double,3> output_ = output.bz<double,3>();
  op(input.bz<T,2>(), output_);
}

static void call_glcm_c(const bob::ip::GLCM& op, bob::python::const_ndarray input, bob::python::ndarray output) 
{
  switch(input.type().dtype) {
    case bob::core::array::t_uint8: return inner_call_glcm<uint8_t>(op, input, output);
    case bob::core::array::t_uint16: return inner_call_glcm<uint16_t>(op, input, output);
    default:
      PYTHON_ERROR(TypeError, "GLCM operator cannot process image of type '%s'", input.type().str().c_str());
  }
}



static object call_glcm_p(const bob::ip::GLCM& op, bob::python::const_ndarray input) 
{
  switch(input.type().dtype) {
    case bob::core::array::t_uint8: 
    {
      const blitz::TinyVector<int,3> sh = op.getGLCMShape(input.bz<uint8_t,2>());
      bob::python::ndarray output(bob::core::array::t_float64, sh(0), sh(1), sh(2));
      inner_call_glcm<uint8_t>(op, input, output);
      return output.self();
    }
    case bob::core::array::t_uint16:
    {
      const blitz::TinyVector<int,3> sh = op.getGLCMShape(input.bz<uint16_t,2>());
      bob::python::ndarray output(bob::core::array::t_float64, sh(0), sh(1), sh(2));
      inner_call_glcm<uint16_t>(op, input, output);
      return output.self();
    }
    default:
      PYTHON_ERROR(TypeError, "GLCM operator cannot process image of type '%s'", input.type().str().c_str());
  }
  return object();
}











void bind_ip_glcm() 
{
  class_<bob::ip::GLCM, boost::shared_ptr<bob::ip::GLCM>, boost::noncopyable>("GLCM", glcm_doc, no_init)
    .def("__init__", make_constructor(&py_constructor_glcm, default_call_policies()),"Constuctor")
    .def(init<const bob::ip::GLCM&>((arg("glcm_operator")), "Copy constructs a GLCM operator"))
    .add_property("offset", &call_get_offset, &call_set_offset, "2D numpy.ndarray of dtype='int32_t' specifying the column and row distance betwee pixel pairs. The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM.")
    .add_property("max_level", &bob::ip::GLCM::getMaxLevel, &bob::ip::GLCM::setMaxLevel, "Grayscale values greater than or equal to this value are scaled to num_levels. The default is the maximum gray-level permitted by the type of input image.")
    .add_property("min_level", &bob::ip::GLCM::getMinLevel, &bob::ip::GLCM::setMinLevel, "Grayscale values smaller than or equal to this value are scaled to 0. The default is the minimum gray-level permitted by the type of input image ")
    .add_property("num_levels", &bob::ip::GLCM::getNumLevels, &bob::ip::GLCM::setNumLevels, "Specifies the number of gray-levels to use when scaling the grayscale values in the input image. This is the number of the values in the first and second dimension in the GLCM matrix. The default is the total number of gray values permitted by the type of the input image")
    .add_property("symmetric", &bob::ip::GLCM::getSymmetric, &bob::ip::GLCM::setSymmetric, " If True, the output matrix for each specified distance and angle will be symmetric. Both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.")
    .add_property("normalized", &bob::ip::GLCM::getNormalized, &bob::ip::GLCM::setNormalized, " If True, each matrix for each specified distance and angle will be normalized by dividing by the total number of accumulated co-occurrences. The default is False.")
    .add_property("round_scaling", &bob::ip::GLCM::getRoundScaling, &bob::ip::GLCM::setRoundScaling, " if True, the quantization (scaling) of the grey-scale values will be done as in Matlab ImageProcessing Toolbox (see method graycomatrix(), http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html) , i.e. with rounding the discrete level. If False, the quantization (scaling) will be done uniformly.")

    .def("__call__", &call_glcm_c, (arg("self"),arg("input"), arg("output")), "Calls an object of this type to extract the GLCM matrix from the given input image.")
    .def("__call__", &call_glcm_p, (arg("self"),arg("input")), "Calls an object of this type to extract the GLCM matrix from the given input image")
    .def("set_quantization_params", &bob::ip::GLCM::setQuantizationParams, (arg("self"),arg("num_levels"), arg("min_level"), arg("max_level")), "Sets all the parameters for quantization (scaling) of the gray levels in one shot.")
    .def("set_glcmtype_params", &bob::ip::GLCM::setGLCMTypeParams, (arg("self"),arg("symmetric"), arg("normalized")), "Sets all the parameters for the type of the output matrix in one shot.")
    .def("get_glcm_shape", &call_get_shape, (arg("self"), arg("input")), "Get the shape of the GLCM matrix goven the input image. It has 3 dimensions: two for the number of grey levels, and one for the number of offsets.")
    //.def("glcm_properties", &call_glcm_properties, (arg("input")), "Calculate a total of 2 so-called Haralick texture properties on the input GLCM matrix. Each column of the resulting numpy.ndarray corresponds to one property, while each row corresponds to one offset of the GLCM matrix. The input is a GLCM matrix in numpy.ndarray format of type 'float64'. The order of the properties is as follows: angular second moment, energy...")
    //.staticmethod("glcm_properties")
    ;
}
