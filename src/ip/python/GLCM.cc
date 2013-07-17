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

#include <bob/python/ndarray.h>
#include <bob/ip/GLCM.h>
#include <boost/make_shared.hpp>

using namespace boost::python;

static const char* glcm_doc = "Objects of this class, after configuration, can compute Grey-Level Co-occurence Matrix of an image.\n This class allows to extract Grey-Level Co-occurence Matrix (GLCM). A thorough tutorial about GLCM and the textural (so-called Haralick) properties that can be derived from it, can be found at: http://www.fp.ucalgary.ca/mhallbey/tutorial.htm \n List of references: \n [1] R. M. Haralick, K. Shanmugam, I. Dinstein; \"Textural Features for Image calssification\", in IEEE Transactions on Systems, Man and Cybernetics, vol.SMC-3, No. 6, p. 610-621. [2] http://www.mathworks.ch/ch/help/images/ref/graycomatrix.html";

template <typename T>  
static void call_set_offset(bob::ip::GLCM<T>& op, bob::python::const_ndarray input)
{
  op.setOffset(input.bz<int32_t,2>());
} 

template <typename T> 
static void call_glcm(const bob::ip::GLCM<T>& op, bob::python::const_ndarray input, bob::python::ndarray output) 
{
  blitz::Array<double,3> output_ = output.bz<double,3>();
  op(input.bz<T,2>(), output_);
}

template <typename T>
static boost::shared_ptr<bob::ip::GLCM<T> > glcm_from_quant(bob::python::const_ndarray quant_thres) {
  return boost::make_shared<bob::ip::GLCM<T> >(quant_thres.bz<T,1>());
}

void bind_ip_glcm_uint8() 
{
  class_<bob::ip::GLCM<uint8_t>, boost::shared_ptr<bob::ip::GLCM<uint8_t>>, boost::noncopyable>("__GLCM_uint8__", glcm_doc, no_init)
    .def(init<>((arg("self")), "Constructor"))
    .def("__init__", make_constructor(&glcm_from_quant<uint8_t>, default_call_policies(), (arg("quantization_table"))), "Constructor")
    .def(init<const int>((arg("self"), arg("num_levels")), "Constructor"))
    .def(init<const int, const uint8_t, const uint8_t>((arg("self"), arg("num_levels"), arg("min_level"), arg("max_level")), "Constructor"))
    .def(init<const bob::ip::GLCM<uint8_t>&>((arg("self"), arg("other")), "Copy constructs a GLCM operator"))
    .add_property("offset", make_function(&bob::ip::GLCM<uint8_t>::getOffset, return_value_policy<copy_const_reference>()), &call_set_offset<uint8_t>, "2D numpy.ndarray of dtype='int32_t' specifying the column and row distance betwee pixel pairs. The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM.")
    .add_property("quantization_table", make_function(&bob::ip::GLCM<uint8_t>::getQuantizationTable, return_value_policy<copy_const_reference>()), "1D numpy.ndarray of dtype='int' containing the thresholds of the quantization. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.")
    .add_property("max_level", &bob::ip::GLCM<uint8_t>::getMaxLevel, "Grayscale values greater than or equal to this value are scaled to num_levels. The default is the maximum gray-level permitted by the type of input image.")
    .add_property("min_level", &bob::ip::GLCM<uint8_t>::getMinLevel, "Grayscale values smaller than or equal to this value are scaled to 0. The default is the minimum gray-level permitted by the type of input image ")
    .add_property("num_levels", &bob::ip::GLCM<uint8_t>::getNumLevels, "Specifies the number of gray-levels to use when scaling the grayscale values in the input image. This is the number of the values in the first and second dimension in the GLCM matrix. The default is the total number of gray values permitted by the type of the input image")
    .add_property("symmetric", &bob::ip::GLCM<uint8_t>::getSymmetric, &bob::ip::GLCM<uint8_t>::setSymmetric, " If True, the output matrix for each specified distance and angle will be symmetric. Both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.")
    .add_property("normalized", &bob::ip::GLCM<uint8_t>::getNormalized, &bob::ip::GLCM<uint8_t>::setNormalized, " If True, each matrix for each specified distance and angle will be normalized by dividing by the total number of accumulated co-occurrences. The default is False.")
    .def("__call__", &call_glcm<uint8_t>, (arg("self"), arg("input"), arg("output")), "Calls an object of this type to extract the GLCM matrix from the given input image.")
    .def("get_glcm_shape", &bob::ip::GLCM<uint8_t>::getGLCMShape, (arg("self")), "Get the shape of the GLCM matrix goven the input image. It has 3 dimensions: two for the number of grey levels, and one for the number of offsets.")
    ;
}

void bind_ip_glcm_uint16() 
{
  class_<bob::ip::GLCM<uint16_t>, boost::shared_ptr<bob::ip::GLCM<uint16_t>>, boost::noncopyable>("__GLCM_uint16__", glcm_doc, no_init)
    .def(init<>((arg("self")), "Constructor"))
    .def("__init__", make_constructor(&glcm_from_quant<uint16_t>, default_call_policies(), (arg("quantization_table"))), "Constructor")
    .def(init<const int>((arg("self"), arg("num_levels")), "Constructor"))
    .def(init<const int, const uint16_t, const uint16_t>((arg("self"), arg("num_levels"), arg("min_level"), arg("max_level")), "Constructor"))
    .def(init<const bob::ip::GLCM<uint16_t>&>((arg("self"), arg("other")), "Copy constructs a GLCM operator"))
    .add_property("offset", make_function(&bob::ip::GLCM<uint16_t>::getOffset, return_value_policy<copy_const_reference>()), &call_set_offset<uint16_t>, "2D numpy.ndarray of dtype='int32_t' specifying the column and row distance betwee pixel pairs. The shape of this array is (num_offsets, 2), where num_offsets is the total number of offsets to be taken into account when computing GLCM.")
    .add_property("quantization_table", make_function(&bob::ip::GLCM<uint16_t>::getQuantizationTable, return_value_policy<copy_const_reference>()), "1D numpy.ndarray of dtype='int' containing the thresholds of the quantization. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.")
    .add_property("max_level", &bob::ip::GLCM<uint16_t>::getMaxLevel, "Grayscale values greater than or equal to this value are scaled to num_levels. The default is the maximum gray-level permitted by the type of input image.")
    .add_property("min_level", &bob::ip::GLCM<uint16_t>::getMinLevel, "Grayscale values smaller than or equal to this value are scaled to 0. The default is the minimum gray-level permitted by the type of input image ")
    .add_property("num_levels", &bob::ip::GLCM<uint16_t>::getNumLevels, "Specifies the number of gray-levels to use when scaling the grayscale values in the input image. This is the number of the values in the first and second dimension in the GLCM matrix. The default is the total number of gray values permitted by the type of the input image")
    .add_property("symmetric", &bob::ip::GLCM<uint16_t>::getSymmetric, &bob::ip::GLCM<uint16_t>::setSymmetric, " If True, the output matrix for each specified distance and angle will be symmetric. Both (i, j) and (j, i) are accumulated when (i, j) is encountered for a given offset. The default is False.")
    .add_property("normalized", &bob::ip::GLCM<uint16_t>::getNormalized, &bob::ip::GLCM<uint16_t>::setNormalized, " If True, each matrix for each specified distance and angle will be normalized by dividing by the total number of accumulated co-occurrences. The default is False.")
    .def("__call__", &call_glcm<uint16_t>, (arg("self"), arg("input"), arg("output")), "Calls an object of this type to extract the GLCM matrix from the given input image.")
    .def("get_glcm_shape", &bob::ip::GLCM<uint16_t>::getGLCMShape, (arg("self")), "Get the shape of the GLCM matrix goven the input image. It has 3 dimensions: two for the number of grey levels, and one for the number of offsets.")
    ;
}

