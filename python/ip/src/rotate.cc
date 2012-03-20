/**
 * @file python/ip/src/Rotate.cc
 * @date Mon Apr 18 11:30:08 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the Rotate class to python
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


#include "ip/rotate.h"
#include "core/python/ndarray.h"
#include <core/array_exception.h>


static boost::python::tuple get_rotated_output_shape(bob::python::const_ndarray input_image, double angle, bool angle_in_degrees){
  // compute angle in degrees, if desired
  if (!angle_in_degrees)
    angle *= 180./M_PI;
  
  // compute output scaling
  blitz::TinyVector<int,2> size;
  
  switch (input_image.type().dtype) {
    case bob::core::array::t_uint8: 
      size = bob::ip::getRotatedShape<uint8_t>(input_image.bz<uint8_t,2>(), angle);
      break;
    case bob::core::array::t_uint16:
      size = bob::ip::getRotatedShape<uint16_t>(input_image.bz<uint16_t,2>(), angle);
      break;
    case bob::core::array::t_float64: 
      size = bob::ip::getRotatedShape<double>(input_image.bz<double,2>(), angle);
      break;
    default: PYTHON_ERROR(TypeError, "cannot get shape from unsupporter array of type '%s'", input_image.type().str().c_str());
  }
   
  return boost::python::make_tuple(size[0], size[1]);
}

template <class T>
  static void inner_rotate(bob::python::const_ndarray input_image, bob::python::ndarray output_image, double angle, const bob::ip::Rotate::Algorithm rotation_algorithm)
{
  
  switch (input_image.type().nd){
    case 2:{
      const blitz::Array<T,2> input = input_image.bz<T,2>();
      blitz::Array<double,2> output = output_image.bz<double,2>();
      bob::ip::rotate(input, output, angle, rotation_algorithm);
      break;
    }
    case 3:{
      const blitz::Array<T,3> input = input_image.bz<T,3>();
      blitz::Array<double,3> output = output_image.bz<double,3>();
      bob::ip::rotate(input, output, angle, rotation_algorithm);
      break;
    }
    default:
      throw bob::core::UnexpectedShapeError();
  }
}

static void rotate(bob::python::const_ndarray input_image, bob::python::ndarray output_image, double angle, bool angle_in_degrees = true, const bob::ip::Rotate::Algorithm rotation_algorithm = bob::ip::Rotate::Shearing){
  // compute angle in degrees, if desired
  if (!angle_in_degrees)
    angle *= 180./M_PI;

  switch (input_image.type().dtype) {
    case bob::core::array::t_uint8:{
      inner_rotate<uint8_t>(input_image, output_image, angle, rotation_algorithm);
      break;
    }
    case bob::core::array::t_uint16:{
      inner_rotate<uint16_t>(input_image, output_image, angle, rotation_algorithm);
      break;
    }
    case bob::core::array::t_float64:{
      inner_rotate<double>(input_image, output_image, angle, rotation_algorithm);
      break;
    }
    default: PYTHON_ERROR(TypeError, "cannot get shape from unsupporter array of type '%s'", input_image.type().str().c_str());
  }
}


template <class T>
  static void inner_rotate_with_mask(bob::python::const_ndarray input_image, const blitz::Array<bool,2>& i_mask, bob::python::ndarray output_image, blitz::Array<bool,2>& o_mask, double angle, const bob::ip::Rotate::Algorithm rotation_algorithm)
{
  
  switch (input_image.type().nd){
    case 2:{
      const blitz::Array<T,2> input = input_image.bz<T,2>();
      blitz::Array<double,2> output = output_image.bz<double,2>();
      bob::ip::rotate(input, output, angle, rotation_algorithm);
      break;
    }
    case 3:{
      const blitz::Array<T,3> input = input_image.bz<T,3>();
      blitz::Array<double,3> output = output_image.bz<double,3>();
      bob::ip::rotate(input, output, angle, rotation_algorithm);
      break;
    }
    default:
      throw bob::core::UnexpectedShapeError();
  }
}


static void rotate_with_mask(bob::python::const_ndarray input_image, bob::python::const_ndarray input_mask, bob::python::ndarray output_image, bob::python::ndarray output_mask, double angle, bool angle_in_degrees = true, const bob::ip::Rotate::Algorithm rotation_algorithm = bob::ip::Rotate::Shearing){
  // compute angle in degrees, if desired
  if (!angle_in_degrees)
    angle *= 180./M_PI;
    
  const blitz::Array<bool,2> i_mask = input_mask.bz<bool,2>();
  blitz::Array<bool,2> o_mask = output_mask.bz<bool,2>();
  
  switch (input_image.type().dtype) {
    case bob::core::array::t_uint8:{
      inner_rotate_with_mask<uint8_t>(input_image, i_mask, output_image, o_mask, angle, rotation_algorithm);
      break;
    }
    case bob::core::array::t_uint16:{
      inner_rotate_with_mask<uint16_t>(input_image, i_mask, output_image, o_mask, angle, rotation_algorithm);
      break;
    }
    case bob::core::array::t_float64:{
      inner_rotate_with_mask<double>(input_image, i_mask, output_image, o_mask, angle, rotation_algorithm);
      break;
    }
    default: PYTHON_ERROR(TypeError, "cannot get shape from unsupporter array of type '%s'", input_image.type().str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(rotate_overloads, rotate, 3, 5)



void bind_ip_rotate() {
  boost::python::enum_<bob::ip::Rotate::Algorithm>("RotateAlgorithm")
    .value("Shearing", bob::ip::Rotate::Shearing)
    .value("BilinearInterp", bob::ip::Rotate::BilinearInterp);
    
  boost::python::def(
    "get_rotated_output_shape", 
    &get_rotated_output_shape, 
    (boost::python::arg("input"), boost::python::arg("angle"), boost::python::arg("angle_in_degrees") = true),
    "Returns the shape of the output image when rotating the given input image with the desired angle. The angle might be given in degree or in radians (please set angle_in_degrees to False in the latter case)."
  );
  
  boost::python::def(
    "rotate",
    &rotate,
    rotate_overloads((boost::python::arg("input"), boost::python::arg("output"), boost::python::arg("angle"), boost::python::arg("angle_in_degrees") = true, boost::python::arg("rotation_algorithm")="Shearing"),
    "Rotates the given input image into the given output image. The size of the output image can be computed using the get_rotated_output_shape function. The angle might be given in degree or in radians (please set angle_in_degrees to False in the latter case).")
  );

  boost::python::def(
    "rotate",
    &rotate_with_mask,
    (boost::python::arg("input"), boost::python::arg("input_mask"), boost::python::arg("output"), boost::python::arg("output_mask"), boost::python::arg("angle"), boost::python::arg("angle_in_degrees") = true, boost::python::arg("rotation_algorithm")="Shearing"),
    "Rotates the given input image into the given output image using the given mask images. The size of the output image and the output mask can be computed using the get_rotated_output_shape function. The angle might be given in degree or in radians (please set angle_in_degrees to False in the latter case)."
  );


  boost::python::def(
    "getAngleToHorizontal", 
    &bob::ip::getAngleToHorizontal, 
    (boost::python::arg("left_h"), boost::python::arg("left_w"), boost::python::arg("right_h"), boost::python::arg("right_w")), 
    "Get the angle needed to level out (horizontally) two points.");
}
