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


#include "bob/ip/rotate.h"
#include "bob/core/python/ndarray.h"
#include "bob/core/array_exception.h"

static boost::python::tuple get_rotated_output_shape(
  bob::python::const_ndarray input, double angle, bool angle_in_degrees)
{
  // Computes angle in degrees, if desired
  if(!angle_in_degrees)
    angle *= 180./M_PI;
  
  // Computes output scaling
  blitz::TinyVector<int,2> size;
  
  switch(input.type().dtype) 
  {
    case bob::core::array::t_uint8: 
      size = bob::ip::getRotatedShape<uint8_t>(input.bz<uint8_t,2>(), angle);
      break;
    case bob::core::array::t_uint16:
      size = bob::ip::getRotatedShape<uint16_t>(input.bz<uint16_t,2>(), angle);
      break;
    case bob::core::array::t_float64: 
      size = bob::ip::getRotatedShape<double>(input.bz<double,2>(), angle);
      break;
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.get_rotated_output_shape() does not support array of type '%s'.", input.type().str().c_str());
  }
   
  return boost::python::make_tuple(size[0], size[1]);
}

template <class T>
static void inner_rotate(bob::python::const_ndarray input, 
  bob::python::ndarray output, double angle, 
  const bob::ip::Rotate::Algorithm rotation_algorithm)
{  
  switch(input.type().nd)
  {
    case 2:
      {
        blitz::Array<double,2> output_ = output.bz<double,2>();
        bob::ip::rotate(input.bz<T,2>(), output_, angle, rotation_algorithm);
        break;
      }
    case 3:
      {
        blitz::Array<double,3> output_ = output.bz<double,3>();
        bob::ip::rotate(input.bz<T,3>(), output_, angle, rotation_algorithm);
        break;
      }
    default:
      PYTHON_ERROR(TypeError, "bob.ip.rotate() does not support array with " SIZE_T_FMT " dimensions.", input.type().nd);
  }
}

static void rotate(bob::python::const_ndarray input, 
  bob::python::ndarray output, double angle, bool angle_in_degrees = true, 
  const bob::ip::Rotate::Algorithm rotation_algorithm = bob::ip::Rotate::Shearing)
{
  // compute angle in degrees, if desired
  if (!angle_in_degrees)
    angle *= 180./M_PI;

  switch(input.type().dtype) 
  {
    case bob::core::array::t_uint8:
      inner_rotate<uint8_t>(input, output, angle, rotation_algorithm);
      break;
    case bob::core::array::t_uint16:
      inner_rotate<uint16_t>(input, output, angle, rotation_algorithm);
      break;
    case bob::core::array::t_float64:
      inner_rotate<double>(input, output, angle, rotation_algorithm);
      break;
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.rotate() does not support array of type '%s'.", input.type().str().c_str());
  }
}


template <class T>
static boost::python::object inner_rotate_p(bob::python::const_ndarray input, 
  double angle, const bob::ip::Rotate::Algorithm rotation_algorithm)
{  
  switch(input.type().nd)
  {
    case 2:
      {
        const blitz::TinyVector<int,2> shape = bob::ip::getRotatedShape<T>(input.bz<T,2>(), angle);
        bob::python::ndarray output(bob::core::array::t_float64, shape(0), shape(1));
        blitz::Array<double,2> output_ = output.bz<double,2>();
        bob::ip::rotate(input.bz<T,2>(), output_, angle, rotation_algorithm);
        return output.self();
      }
    case 3:
      {
        const blitz::TinyVector<int,3> shape = bob::ip::getRotatedShape<T>(input.bz<T,3>(), angle);
        bob::python::ndarray output(bob::core::array::t_float64, shape(0), shape(1), shape(2));
        blitz::Array<double,3> output_ = output.bz<double,3>();
        bob::ip::rotate(input.bz<T,3>(), output_, angle, rotation_algorithm);
        return output.self();
      }
    default:
      PYTHON_ERROR(TypeError, "bob.ip.rotate() does not support array with " SIZE_T_FMT " dimensions.", input.type().nd);
  }
}

static boost::python::object rotate_p(bob::python::const_ndarray input, 
  double angle, bool angle_in_degrees = true, 
  const bob::ip::Rotate::Algorithm rotation_algorithm = bob::ip::Rotate::Shearing)
{
  // compute angle in degrees, if desired
  if (!angle_in_degrees)
    angle *= 180./M_PI;

  switch(input.type().dtype) 
  {
    case bob::core::array::t_uint8:
      return inner_rotate_p<uint8_t>(input, angle, rotation_algorithm);
    case bob::core::array::t_uint16:
      return inner_rotate_p<uint16_t>(input, angle, rotation_algorithm);
    case bob::core::array::t_float64:
      return inner_rotate_p<double>(input, angle, rotation_algorithm);
    default: 
      PYTHON_ERROR(TypeError, "bob.ip.rotate() does not support array of type '%s'.", input.type().str().c_str());
  }
}

template <class T>
static void inner_rotate_with_mask(bob::python::const_ndarray input, 
  const blitz::Array<bool,2>& i_mask, bob::python::ndarray output, 
  blitz::Array<bool,2>& o_mask, double angle, 
  const bob::ip::Rotate::Algorithm rotation_algorithm)
{
  switch (input.type().nd)
  {
    case 2:
      {
        blitz::Array<double,2> output_ = output.bz<double,2>();
        bob::ip::rotate(input.bz<T,2>(), output_, angle, rotation_algorithm);
        break;
      }
    case 3:
      {
        blitz::Array<double,3> output_ = output.bz<double,3>();
        bob::ip::rotate(input.bz<T,3>(), output_, angle, rotation_algorithm);
        break;
      }
    default:
      PYTHON_ERROR(TypeError, "bob.ip.rotate() does not support array with " SIZE_T_FMT " dimensions.", input.type().nd);
  }
}


static void rotate_with_mask(bob::python::const_ndarray input, 
  bob::python::const_ndarray input_mask, bob::python::ndarray output, 
  bob::python::ndarray output_mask, double angle, bool angle_in_degrees=true, 
  const bob::ip::Rotate::Algorithm rotation_algorithm = bob::ip::Rotate::Shearing)
{
  // computes angle in degrees, if desired
  if (!angle_in_degrees)
    angle *= 180./M_PI;
    
  const blitz::Array<bool,2> i_mask = input_mask.bz<bool,2>();
  blitz::Array<bool,2> o_mask = output_mask.bz<bool,2>();
  
  switch (input.type().dtype) 
  {
    case bob::core::array::t_uint8:
      inner_rotate_with_mask<uint8_t>(input, i_mask, output, o_mask, angle, rotation_algorithm);
      break;
    case bob::core::array::t_uint16:
      inner_rotate_with_mask<uint16_t>(input, i_mask, output, o_mask, angle, rotation_algorithm);
      break;
    case bob::core::array::t_float64:
      inner_rotate_with_mask<double>(input, i_mask, output, o_mask, angle, rotation_algorithm);
      break;
    default:
      PYTHON_ERROR(TypeError, "bob.ip.rotate() does not support array of type '%s'.", input.type().str().c_str());
  }
}

BOOST_PYTHON_FUNCTION_OVERLOADS(rotate_overloads, rotate, 3, 5)

BOOST_PYTHON_FUNCTION_OVERLOADS(rotate_p_overloads, rotate_p, 2, 4)

BOOST_PYTHON_FUNCTION_OVERLOADS(rotate_with_mask_overloads, rotate_with_mask, 5, 7)

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
    rotate_overloads(
      (boost::python::arg("input"), boost::python::arg("output"), boost::python::arg("angle"), boost::python::arg("angle_in_degrees") = true, boost::python::arg("rotation_algorithm")="Shearing"),
      "Rotates the given input image into the given output image. The ouput image should have the expected size, which can be obtained using the get_rotated_output_shape function. The angle might be given in degree or in radians (please set angle_in_degrees to False in the latter case)."
    )
  );
 
  boost::python::def(
    "rotate",
    &rotate_p,
    rotate_p_overloads(
      (boost::python::arg("input"), boost::python::arg("angle"), boost::python::arg("angle_in_degrees") = true, boost::python::arg("rotation_algorithm")="Shearing"),
      "Rotates the given input image into the given output image. The angle might be given in degree or in radians (please set angle_in_degrees to False in the latter case). The rotated image is allocated and returned."
    )
  );

  boost::python::def(
    "rotate",
    &rotate_with_mask,
    rotate_with_mask_overloads(
      (boost::python::arg("input"), boost::python::arg("input_mask"), boost::python::arg("output"), boost::python::arg("output_mask"), boost::python::arg("angle"), boost::python::arg("angle_in_degrees") = true, boost::python::arg("rotation_algorithm")="Shearing"),
      "Rotates the given input image into the given output image using the given mask images. The size of the output image and the output mask can be computed using the get_rotated_output_shape function. The angle might be given in degree or in radians (please set angle_in_degrees to False in the latter case)."
    )
  );

  boost::python::def(
    "get_angle_to_horizontal", 
    &bob::ip::getAngleToHorizontal, 
    (boost::python::arg("left_y"), boost::python::arg("left_x"), boost::python::arg("right_y"), boost::python::arg("right_x")), 
    "Get the angle needed to level out (horizontally) two points.");
}
