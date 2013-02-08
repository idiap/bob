/**
 * @file ip/python/Quantization.cc
 * @date Wed Feb  6 10:24:57 CET 2013
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 *
 * @brief Binds the Quantization class to python
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

#include "bob/sp/Quantization.h"

#include "bob/core/python/ndarray.h"
#include <boost/make_shared.hpp>

using namespace boost::python;

static const char* quantization_doc = "Objects of this class, after configuration, can quantize 1D or 2D signal into different number of levels. At the moment, only uint8 and uint16 input signals are supported.";


/*static boost::shared_ptr<bob::sp::Quantization<uint8_t>> py_constructor_quantization_uint8() 
{  
  return boost::make_shared<bob::sp::Quantization<uint8_t>>();
}*/
  
    
template <typename T> 
static object call_get_thresholds(const bob::sp::Quantization<T>& op)
{
  blitz::Array<T,1> output=op.getThresholds();
  return object(output);
}  




template <typename T, int N> 
static void inner_call_quantization(const bob::sp::Quantization<T>& op, bob::python::const_ndarray input, bob::python::ndarray output) 
{
  blitz::Array<uint32_t,N> output_ = output.bz<uint32_t,N>();
  op(input.bz<T,N>(), output_);
}

template <typename T>
static void call_quantization_c(const bob::sp::Quantization<T>& op, bob::python::const_ndarray input, bob::python::ndarray output) 
{
  const bob::core::array::typeinfo& input_type = input.type();
  //switch(input.type().dtype) {
  
    //case bob::core::array::t_uint8: 
  if (input_type.nd == 1)
    return inner_call_quantization<T, 1>(op, input, output);
  if (input_type.nd == 2)
    return inner_call_quantization<T, 2>(op, input, output);   
}


void bind_sp_quantization_uint8() 
{
  class_<bob::sp::Quantization<uint8_t>, boost::shared_ptr<bob::sp::Quantization<uint8_t>>, boost::noncopyable>("__quantization_uint8__", quantization_doc, no_init)
    .def(init<>("Constructor. "))
    .def(init<const int, const int>((arg("quantization_type"), arg("num_levels")),"Constructor"))
    .def(init<const int, const int, const uint8_t, const uint8_t>((arg("quantization_type"), arg("num_levels"), arg("min_level"), arg("max_level")),"Constructor"))
    .def(init<const blitz::Array<uint8_t,1>&>((arg("quantization_table")),"Constructor"))
    
    .def(init<const bob::sp::Quantization<uint8_t>&>((arg("quantization_operator")), "Copy constructs a Quantization operator"))
    
    .add_property("thresholds", &call_get_thresholds<uint8_t>, "1D numpy.ndarray of dtype='int' containing the thresholds of the quantization. Should be pre-given if the the type of quantization m_type=3. Otherwise, it is recalculated according to other quantization parameters. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.")
    .add_property("max_level", &bob::sp::Quantization<uint8_t>::getMaxLevel, "Input values greater then this value are scaled to this value prior to quantization. As a result, they will be quantized in the highest quantization level. The default is the maximum value permitted by the data type of the class.")
    .add_property("min_level", &bob::sp::Quantization<uint8_t>::getMinLevel, "Input values smaller than or equal to this value are scaled to this value prior to quantization. As a result, they will be scaled in the lowest qunatization level. The default is the minimum value permitted by the data type. ")
    .add_property("num_levels", &bob::sp::Quantization<uint8_t>::getNumLevels, "Number of quantization levels. The default is the total number of discreet values permitted by the data type of the class.")
    .add_property("type", &bob::sp::Quantization<uint8_t>::getType, "Type of quantization: 1 - uniform quantization, 2 - uniform quantization with rounding, 3 - quantization by user-specified quantization table. Default is 1.")
    
    .def("__call__", &call_quantization_c<uint8_t>, (arg("self"),arg("input"), arg("output")), "Calls an object of this type to perform quantization for the input signal.")
    
    .def("quantization_level", &bob::sp::Quantization<uint8_t>::quantization_level, (arg("self"),arg("input")), "Calculates the quantization level for a single input value, given the current thresholds table.")
    ;
}

void bind_sp_quantization_uint16() 
{
  class_<bob::sp::Quantization<uint16_t>, boost::shared_ptr<bob::sp::Quantization<uint16_t>>, boost::noncopyable>("__quantization_uint16__", quantization_doc, no_init)
    .def(init<>("Constructor"))
    .def(init<const int, const int>((arg("quantization_type"), arg("num_levels")),"Constructor"))
    .def(init<const int, const int, const uint16_t, const uint16_t>((arg("quantization_type"), arg("num_levels"), arg("min_level"), arg("max_level")),"Constructor"))
    .def(init<const blitz::Array<uint16_t,1>&>((arg("quantization_table")),"Constructor"))
    
    .def(init<const bob::sp::Quantization<uint16_t>&>((arg("quantization_operator")), "Copy constructs a Quantization operator"))
    
    .add_property("thresholds", &call_get_thresholds<uint16_t>, "1D numpy.ndarray of dtype='int' containing the thresholds of the quantization. Should be pre-given if the the type of quantization m_type=3. Otherwise, it is recalculated according to other quantization parameters. Each element corresponds to the lower boundary of the particular quantization level. Eg. array([ 0,  5, 10]) means quantization in 3 levels. Input values in the range [0,4] will be quantized to level 0, input values in the range[5,9] will be quantized to level 1 and input values in the range [10-max_level] will be quantized to level 2.")
    .add_property("max_level", &bob::sp::Quantization<uint16_t>::getMaxLevel, "Input values greater then this value are scaled to this value prior to quantization. As a result, they will be quantized in the highest quantization level. The default is the maximum value permitted by the data type of the class.")
    .add_property("min_level", &bob::sp::Quantization<uint16_t>::getMinLevel, "Input values smaller than or equal to this value are scaled to this value prior to quantization. As a result, they will be scaled in the lowest qunatization level. The default is the minimum value permitted by the data type. ")
    .add_property("num_levels", &bob::sp::Quantization<uint16_t>::getNumLevels, "Number of quantization levels. The default is the total number of discreet values permitted by the data type of the class.")
    .add_property("type", &bob::sp::Quantization<uint16_t>::getType, "Type of quantization: 1 - uniform quantization, 2 - uniform quantization with rounding, 3 - quantization by user-specified quantization table. Default is 1.")
    
    .def("__call__", &call_quantization_c<uint16_t>, (arg("self"),arg("input"), arg("output")), "Calls an object of this type to perform quantization for the input signal.")
    
    .def("quantization_level", &bob::sp::Quantization<uint16_t>::quantization_level, (arg("self"),arg("input")), "Calculates the quantization level for a single input value, given the current thresholds table.")
    ;
}
