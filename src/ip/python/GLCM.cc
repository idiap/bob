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

#include "bob/python/ndarray.h"
#include <boost/make_shared.hpp>

using namespace boost::python;

static const char* glcm_doc = "Objects of this class, after configuration, can compute Grey-Level Co-occurence Matrix of an image.\n This class allows to extract Grey-Level Co-occurence Matrix (GLCM). A thorough tutorial about GLCM and the textural (so-called Haralick) properties that can be derived from it, can be found at: http://www.fp.ucalgary.ca/mhallbey/tutorial.htm \n List of references: \n [1] R. M. Haralick, K. Shanmugam, I. Dinstein; \"Textural Features for Image calssification\", in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621. [2] http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html";

/*
static boost::shared_ptr<bob::ip::GLCM> py_constructor_glcm() 
{  
  return boost::make_shared<bob::ip::GLCM>();
}*/
  
template <typename T>  
static void call_set_offset(bob::ip::GLCM<T>& op, bob::python::const_ndarray input)
{
  op.setOffset(input.bz<int32_t,2>());
} 

template <typename T>
static object call_get_offset(const bob::ip::GLCM<T>& op)
{
  blitz::Array<int32_t,2> output=op.getOffset();
  return object(output);
}  

template <typename T>
static object call_get_quantization_table(const bob::ip::GLCM<T>& op)
{
  blitz::Array<T,1> output=op.getQuantizationTable();
  return object(output);
}  
      
template <typename T>
static const blitz::TinyVector<int,3> call_get_shape(const bob::ip::GLCM<T>& op)
{
  return op.getGLCMShape();
} 


template <typename T> 
static void call_glcm(const bob::ip::GLCM<T>& op, bob::python::const_ndarray input, bob::python::ndarray output) 
{
  blitz::Array<double,3> output_ = output.bz<double,3>();
  op(input.bz<T,2>(), output_);
}










void bind_ip_glcm_uint8() 
{
  class_<bob::ip::GLCM<uint8_t>, boost::shared_ptr<bob::ip::GLCM<uint8_t>>, boost::noncopyable>("__GLCM_uint8__", glcm_doc, no_init)

    .def(init<>("Constructor. "))
    .def(init<const int>((arg("num_levels")),"Constructor"))
    .def(init<const int, const uint8_t, const uint8_t>((arg("num_levels"), arg("min_level"), arg("max_level")),"Constructor"))
    .def(init<const blitz::Array<uint8_t,1>&>((arg("quantization_table")),"Constructor")) 

    .def(init<const bob::ip::GLCM<uint8_t>&>((arg("glcm_operator")), "Copy constructs a GLCM operator"))
    
    .add_property("offset", &call_get_offset<uint8_t>, &call_set_offset<uint8_t>, "2D numpy.ndarray of dtype='int32_t' specifying the column and row distance betwee pixel pairs. The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM.")
    .add_property("quantization_table", &call_get_quantization_table<uint8_t>, "1D numpy.ndarray of dtype='int' containing the thresholds of the quantization. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.")
    .add_property("max_level", &bob::ip::GLCM<uint8_t>::getMaxLevel, "Grayscale values greater than or equal to this value are scaled to num_levels. The default is the maximum gray-level permitted by the type of input image.")
    .add_property("min_level", &bob::ip::GLCM<uint8_t>::getMinLevel, "Grayscale values smaller than or equal to this value are scaled to 0. The default is the minimum gray-level permitted by the type of input image ")
    .add_property("num_levels", &bob::ip::GLCM<uint8_t>::getNumLevels, "Specifies the number of gray-levels to use when scaling the grayscale values in the input image. This is the number of the values in the first and second dimension in the GLCM matrix. The default is the total number of gray values permitted by the type of the input image")
    .add_property("symmetric", &bob::ip::GLCM<uint8_t>::getSymmetric, &bob::ip::GLCM<uint8_t>::setSymmetric, " If True, the output matrix for each specified distance and angle will be symmetric. Both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.")
    .add_property("normalized", &bob::ip::GLCM<uint8_t>::getNormalized, &bob::ip::GLCM<uint8_t>::setNormalized, " If True, each matrix for each specified distance and angle will be normalized by dividing by the total number of accumulated co-occurrences. The default is False.")

    .def("__call__", &call_glcm<uint8_t>, (arg("self"),arg("input"), arg("output")), "Calls an object of this type to extract the GLCM matrix from the given input image.")
    .def("get_glcm_shape", &call_get_shape<uint8_t>, (arg("self")), "Get the shape of the GLCM matrix goven the input image. It has 3 dimensions: two for the number of grey levels, and one for the number of offsets.")
    //.def("glcm_properties", &call_glcm_properties, (arg("input")), "Calculate a total of 2 so-called Haralick texture properties on the input GLCM matrix. Each column of the resulting numpy.ndarray corresponds to one property, while each row corresponds to one offset of the GLCM matrix. The input is a GLCM matrix in numpy.ndarray format of type 'float64'. The order of the properties is as follows: angular second moment, energy...")
    //.staticmethod("glcm_properties")
    ;
}



void bind_ip_glcm_uint16() 
{
  class_<bob::ip::GLCM<uint16_t>, boost::shared_ptr<bob::ip::GLCM<uint16_t>>, boost::noncopyable>("__GLCM_uint16__", glcm_doc, no_init)

    .def(init<>("Constructor. "))
    .def(init<const int>((arg("num_levels")),"Constructor"))
    .def(init<const int, const uint16_t, const uint16_t>((arg("num_levels"), arg("min_level"), arg("max_level")),"Constructor"))
    .def(init<const blitz::Array<uint16_t,1>&>((arg("quantization_table")),"Constructor")) 

    .def(init<const bob::ip::GLCM<uint16_t>&>((arg("glcm_operator")), "Copy constructs a GLCM operator"))
    
    .add_property("offset", &call_get_offset<uint16_t>, &call_set_offset<uint16_t>, "2D numpy.ndarray of dtype='int32_t' specifying the column and row distance betwee pixel pairs. The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM.")
    .add_property("quantization_table", &call_get_quantization_table<uint16_t>, "1D numpy.ndarray of dtype='int' containing the thresholds of the quantization. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.")
    .add_property("max_level", &bob::ip::GLCM<uint16_t>::getMaxLevel, "Grayscale values greater than or equal to this value are scaled to num_levels. The default is the maximum gray-level permitted by the type of input image.")
    .add_property("min_level", &bob::ip::GLCM<uint16_t>::getMinLevel, "Grayscale values smaller than or equal to this value are scaled to 0. The default is the minimum gray-level permitted by the type of input image ")
    .add_property("num_levels", &bob::ip::GLCM<uint16_t>::getNumLevels, "Specifies the number of gray-levels to use when scaling the grayscale values in the input image. This is the number of the values in the first and second dimension in the GLCM matrix. The default is the total number of gray values permitted by the type of the input image")
    .add_property("symmetric", &bob::ip::GLCM<uint16_t>::getSymmetric, &bob::ip::GLCM<uint16_t>::setSymmetric, " If True, the output matrix for each specified distance and angle will be symmetric. Both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.")
    .add_property("normalized", &bob::ip::GLCM<uint16_t>::getNormalized, &bob::ip::GLCM<uint16_t>::setNormalized, " If True, each matrix for each specified distance and angle will be normalized by dividing by the total number of accumulated co-occurrences. The default is False.")

    .def("__call__", &call_glcm<uint16_t>, (arg("self"),arg("input"), arg("output")), "Calls an object of this type to extract the GLCM matrix from the given input image.")
    .def("get_glcm_shape", &call_get_shape<uint16_t>, (arg("self")), "Get the shape of the GLCM matrix goven the input image. It has 3 dimensions: two for the number of grey levels, and one for the number of offsets.")
    //.def("glcm_properties", &call_glcm_properties, (arg("input")), "Calculate a total of 2 so-called Haralick texture properties on the input GLCM matrix. Each column of the resulting numpy.ndarray corresponds to one property, while each row corresponds to one offset of the GLCM matrix. The input is a GLCM matrix in numpy.ndarray format of type 'float64'. The order of the properties is as follows: angular second moment, energy...")
    //.staticmethod("glcm_properties")
    ;
}
