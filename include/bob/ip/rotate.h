/**
 * @file bob/ip/rotate.h
 * @date Sun Apr 17 23:11:51 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to rotate a 2D or 3D array/image.
 * The shearing-based algorithm is strongly inspired by the following article:
 * 'A Fast Algorithm for General Raster Rotation', Alan Paeth, in the
 * proceedings of Graphics Interface '86, p. 77-81.
 * The notes of Tobin Fricke about this article might also be of interest.
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

#ifndef BOB_IP_ROTATE_H
#define BOB_IP_ROTATE_H

#include <blitz/array.h>
#include "bob/core/array_assert.h"
#include "bob/core/cast.h"
#include "bob/ip/Exception.h"
#include "bob/ip/shear.h"
#include "bob/ip/crop.h"


namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {
    namespace detail {
    }
    
    namespace Rotate {
      /**
        * @brief Internal enumeration of the possible algorithms
        */
      typedef enum Algorithm_ {
        Shearing,
        BilinearInterp
      } Algorithm;

    }
    
    //! Returns the shape the given image will be rotated into when using the specified angle in degrees.
    template<typename T>
      static const blitz::TinyVector<int,2> getRotatedShape( 
        const blitz::Array<T,2>& src, 
        const double angle
      );
      
    //! Returns the shape the given image will be rotated into when using the specified angle in degrees.
    template<typename T>
      static const blitz::TinyVector<int,3> getRotatedShape( 
        const blitz::Array<T,3>& src, 
        const double angle
      );
      
    
    //! Rotates the given input image to the given output image with the given angle in degrees.
    template <class T>
      void rotate(
        const blitz::Array<T,2>& src,
        blitz::Array<double,2>& dst,
        const double angle,
        const bob::ip::Rotate::Algorithm algorithm = bob::ip::Rotate::Shearing
      );
      
    //! Rotates the given input image to the given output image 
    template <class T>
      void rotate(
        const blitz::Array<T,2>& src,
        const blitz::Array<bool,2>& src_mask,
        blitz::Array<double,2>& dst,
        blitz::Array<bool,2>& dst_mask,
        const double angle,
        const bob::ip::Rotate::Algorithm algorithm = bob::ip::Rotate::Shearing
      );


    //! Rotates the given input image to the given output image with the given angle in degrees.
    template <class T>
      void rotate(
        const blitz::Array<T,3>& src,
        blitz::Array<double,3>& dst,
        const double angle,
        const bob::ip::Rotate::Algorithm algorithm = bob::ip::Rotate::Shearing
      );
      
    //! Rotates the given input image to the given output image using the specified mask.
    template <class T>
      void rotate(
        const blitz::Array<T,3>& src,
        const blitz::Array<bool,2>& src_mask,
        blitz::Array<double,3>& dst,
        blitz::Array<bool,2>& dst_mask,
        const double angle,
        const bob::ip::Rotate::Algorithm algorithm = bob::ip::Rotate::Shearing
      );

    //! Computes the angle between the line of the two given points and the horizontal line
    double getAngleToHorizontal(
      const double left_y, 
      const double left_x,
      const double right_y, 
      const double right_x
    );
    

  } // namespace ip$
/**
 * @}
 */
} // namespace bob


//////////////////////////////////////////////////////////////////////////////////////
//// internal implementations of the rotate functions. ///////////////////////////////


/**
  * @brief Function which rotates a 2D blitz::array/image of a given type
  *   with an angle of 0 degrees.
  * @warning No check is performed on the dst blitz::array/image.
  * @param src The input blitz array
  * @param src_mask The mask for the input image
  * @param dst The output blitz array
  * @param dst_mask The mask for the output image (that will be generated)
  */
template<typename T, bool mask>
  static inline void rotateNoCheck_0(
    const blitz::Array<T,2>& src, 
    const blitz::Array<bool,2>& src_mask, 
    blitz::Array<double,2>& dst,
    blitz::Array<bool,2>& dst_mask
)
{ 
  for( int y=0; y<dst.extent(0); ++y)
    for( int x=0; x<dst.extent(1); ++x)
      dst(y,x) = bob::core::cast<double>(src(y,x));
  if(mask) {
    for( int y=0; y<dst.extent(0); ++y)
      for( int x=0; x<dst.extent(1); ++x)
        dst_mask(y,x) = bob::core::cast<double>(src_mask(y,x));
  }
}

/**
  * @brief Function which rotates a 2D blitz::array/image of a given type
  *   with an angle of 90 degrees.
  * @warning No check is performed on the dst blitz::array/image.
  * @param src The input blitz array
  * @param src_mask The mask for the input image
  * @param dst The output blitz array
  * @param dst_mask The mask for the output image (that will be generated)
  */
template<typename T, bool mask>
  static inline void rotateNoCheck_90(
    const blitz::Array<T,2>& src, 
    const blitz::Array<bool,2>& src_mask, 
    blitz::Array<double,2>& dst,
    blitz::Array<bool,2>& dst_mask
)
{ 
  for( int y=0; y<dst.extent(0); ++y)
    for( int x=0; x<dst.extent(1); ++x)
      dst(y,x) = static_cast<double>(src( x, (src.extent(1)-1-y) ));
  if(mask) {
    for( int y=0; y<dst.extent(0); ++y)
      for( int x=0; x<dst.extent(1); ++x)
        dst_mask(y,x) = bob::core::cast<double>(
                          src_mask(x, (src.extent(1)-1-y) ));
  }
}

/**
  * @brief Function which rotates a 2D blitz::array/image of a given type
  *   with an angle of 180 degrees.
  * @warning No check is performed on the dst blitz::array/image.
  * @param src The input blitz array
  * @param src_mask The mask for the input image
  * @param dst The output blitz array
  * @param dst_mask The mask for the output image (that will be generated)
  */
template<typename T, bool mask>
  static inline void rotateNoCheck_180(
  const blitz::Array<T,2>& src, 
  const blitz::Array<bool,2>& src_mask, 
  blitz::Array<double,2>& dst,
  blitz::Array<bool,2>& dst_mask
)
{ 
  for( int y=0; y<dst.extent(0); ++y)
    for( int x=0; x<dst.extent(1); ++x)
      dst(y,x) = bob::core::cast<double>(src( (src.extent(0)-1-y),
                                          (src.extent(1)-1-x) ));
  if(mask) {
    for( int y=0; y<dst.extent(0); ++y)
      for( int x=0; x<dst.extent(1); ++x)
        dst_mask(y,x) = bob::core::cast<double>(
                          src_mask( (src.extent(0)-1-y), 
                                    (src.extent(1)-1-x) ));
  }
}

/**
  * @brief Function which rotates a 2D blitz::array/image of a given type
  *   with an angle of 270 degrees.
  * @warning No check is performed on the dst blitz::array/image.
  * @param src The input blitz array
  * @param src_mask The mask for the input image
  * @param dst The output blitz array
  * @param dst_mask The mask for the output image (that will be generated)
  */
template<typename T, bool mask>
  static inline void rotateNoCheck_270(
  const blitz::Array<T,2>& src,
  const blitz::Array<bool,2>& src_mask, 
  blitz::Array<double,2>& dst,
  blitz::Array<bool,2>& dst_mask
)
{ 
  for( int y=0; y<dst.extent(0); ++y)
    for( int x=0; x<dst.extent(1); ++x)
      dst(y,x) = bob::core::cast<double>(src( (src.extent(0)-1-x), y ));
  if(mask) {
    for( int y=0; y<dst.extent(0); ++y)
      for( int x=0; x<dst.extent(1); ++x)
        dst_mask(y,x) = bob::core::cast<double>(
                          src_mask( (src.extent(0)-1-x), y ));
  }
}


/**
  * @brief Function which rotates a 2D blitz::array/image using a
  *   shearing algorithm.
  * @warning No check is performed on the dst blitz::array/image.
  * @param src The input blitz array
  * @param src_mask The mask for the input image
  * @param dst The output blitz array
  * @param dst_mask The mask for the output image (that will be generated)
  * @param angle The angle of the rotation (in degrees)
  */
template<typename T, bool mask>
  static inline void rotateShearingNoCheck(
    const blitz::Array<T,2>& src, 
    const blitz::Array<bool,2>& src_mask, 
    blitz::Array<double,2>& dst,
    blitz::Array<bool,2>& dst_mask, 
    const double angle
  )
{ 
  // Determine the quadrant of the original angle:
  //   0:[-45,45] -- 1:[45,135] -- 2:[135,225] -- 3:[225,315(i.e. -45)]
  double angle_norm = angle;
  size_t quadrant;
  for( quadrant=0; angle_norm > 45.; ++quadrant)
    angle_norm -= 90.;
  quadrant %= 4;

  // Compute useful values 
  double rad_angle = angle_norm * M_PI / 180.;

  // declare temporary arrays used for rotation
  blitz::Array<double,2> dst_int1, dst_int2, dst_int3, dst_int4;
  blitz::Array<bool,2> mask_int1, mask_int2, mask_int3, mask_int4;

  // Declare an intermediate arrays
  // Perform simple rotation. After that, there is one more
  //  rotation to do with an angle in [-45,45].
  if( quadrant == 0 ) {
    dst_int1.resize( src.extent(0), src.extent(1) );
    if(mask)
      mask_int1.resize( src.extent(0), src.extent(1) );
    rotateNoCheck_0<T,mask>(src, src_mask, dst_int1, mask_int1);
  }
  else if( quadrant == 1) {
    dst_int1.resize( src.extent(1), src.extent(0) );
    if(mask)
      mask_int1.resize( src.extent(1), src.extent(0) );
    rotateNoCheck_90<T,mask>(src, src_mask, dst_int1, mask_int1);
  }
  else if( quadrant == 2) {
    dst_int1.resize( src.extent(0), src.extent(1) );
    if(mask)
      mask_int1.resize( src.extent(0), src.extent(1) );
    rotateNoCheck_180<T,mask>(src, src_mask, dst_int1, mask_int1);
  }
  else { // quadrant == 3
    dst_int1.resize( src.extent(1), src.extent(0) );
    if(mask)
      mask_int1.resize( src.extent(1), src.extent(0) );
    rotateNoCheck_270<T,mask>(src, src_mask, dst_int1, mask_int1);
  }

  // Compute shearing values required for the rotation
  const double shear_x = -tan( rad_angle / 2. );
  const double shear_y = sin( rad_angle );

  // Perform first shear (shearX)
  const blitz::TinyVector<int,2> s1 = bob::ip::getShearXShape(dst_int1, shear_x);
  dst_int2.resize(s1);
  if(mask){
    mask_int2.resize( s1 );
    bob::ip::shearX( dst_int1, mask_int1, dst_int2, mask_int2, shear_x, true);
  }else 
    bob::ip::shearX( dst_int1, dst_int2, shear_x, true);
    
  // Perform second shear (shearY)
  const blitz::TinyVector<int,2> s2 = bob::ip::getShearYShape(dst_int2, shear_y);
  dst_int3.resize(s2);
  if(mask){
    mask_int3.resize( s2 );
    bob::ip::shearY( dst_int2, mask_int2, dst_int3, mask_int3, shear_y, true);
  }else
    bob::ip::shearY( dst_int2, dst_int3, shear_y, true);
    
  // Perform third shear (shearX)
  const blitz::TinyVector<int,2> s3 = bob::ip::getShearXShape(dst_int3, shear_x);
  dst_int4.resize(s3);
  if(mask){
    mask_int4.resize( s3 );
    bob::ip::shearX( dst_int3, mask_int3, dst_int4, mask_int4, shear_x, true);
  }else
    bob::ip::shearX( dst_int3, dst_int4, shear_x, true);

  // Crop obtained sheared image
  const blitz::TinyVector<int,2> crop_d = bob::ip::getRotatedShape(src, angle);

  int crop_x = (dst_int4.extent(1) - crop_d(1)) / 2;
  int crop_y = (dst_int4.extent(0) - crop_d(0)) / 2;
  if(mask)
    bob::ip::crop( dst_int4, mask_int4, dst, dst_mask, crop_y, crop_x,
        crop_d(0), crop_d(1), true, true);
  else 
    bob::ip::crop( dst_int4, dst, crop_y, crop_x, crop_d(0), crop_d(1),
        true, true);
}


/**
  * @brief Function which rotates a 2D blitz::array/image 
  * @warning No check is performed on the dst blitz::array/image.
  * @param src The input blitz array
  * @param src_mask The mask for the input image
  * @param dst The output blitz array
  * @param dst_mask The mask for the output image (that will be generated)
  * @param angle The angle of the rotation (in degrees)
  * @param algo The algorithm which should be used to perform the 
  *   rotation.
  */
template<typename T, bool mask>
  static inline void rotateNoCheck(
    const blitz::Array<T,2>& src, 
    const blitz::Array<bool,2>& src_mask, 
    blitz::Array<double,2>& dst,
    blitz::Array<bool,2>& dst_mask,
    const double angle,
    const bob::ip::Rotate::Algorithm algo
  )
{ 
  // Force the angle to be in range [-45,315] 
  double angle_norm = angle;
  while(angle_norm < -45.)
    angle_norm += 360.;
  while(angle_norm > 315.)
    angle_norm -= 360.;

  // Check and resize dst if required
  if(angle_norm == 0. || angle_norm == 180.)
  {
    // Perform rotation
    if(angle_norm == 0.)
      rotateNoCheck_0<T,mask>(src, src_mask, dst, dst_mask);
    else
      rotateNoCheck_180<T,mask>(src, src_mask, dst, dst_mask);
    return;
  }
  else if(angle_norm == 90. || angle_norm == 270.)
  {
    // Perform rotation
    if(angle_norm == 90.)
      rotateNoCheck_90<T,mask>(src, src_mask, dst, dst_mask);
    else
      rotateNoCheck_270<T,mask>(src, src_mask, dst, dst_mask);
    return;
  }

  switch(algo)
  {
    case bob::ip::Rotate::Shearing:
      rotateShearingNoCheck<T,mask>(src, src_mask, dst, dst_mask, angle_norm);
      break;
    default:
      throw bob::ip::UnknownRotatingAlgorithm();
  }
}



//////////////////////////////////////////////////////////////////////////////////////
//// public API of the rotate functions. /////////////////////////////////////////////

/**
  * Rotate the given 2D input image to the given output image. 
  * The size of the output image must be identical to the size returned by bob::ip::getRotatedShape(src,angle).
  * @param src  The source image to rotate.
  * @param dst  The destination image that will hold the rotated image
  * @param angle  The angle in degrees the input image is rotated
  * @param algo  The algorithm used for image interpolation.
*/
template <typename T> 
  inline void bob::ip::rotate(
    const blitz::Array<T,2>& src, 
    blitz::Array<double,2>& dst,
    const double angle,
    const bob::ip::Rotate::Algorithm algo
  )
{
  // Check input
  bob::core::array::assertZeroBase(src);

  // Check output
  bob::core::array::assertZeroBase(dst);
  bob::core::array::assertSameShape(dst, bob::ip::getRotatedShape(src,angle));

  // Perform the rotation
  blitz::Array<bool,2> src_mask, dst_mask;
  rotateNoCheck<T,false>(src, src_mask, dst, dst_mask, angle, algo);
}


/**
  * Rotate the given 2D input image to the given output image using the specified input mask.
  * The size of the output image and the output mask must be identical to the size returned by bob::ip::getRotatedShape(src,angle).
  * @param src  The source image to rotate.
  * @param src_mask  The mask that should be applied
  * @param dst  The destination image that will hold the rotated image
  * @param dst_mask  The mask of the destination image
  * @param angle  The angle in degrees the input image is rotated
  * @param algo  The algorithm used for image interpolation.
*/
template <typename T> 
  inline void bob::ip::rotate(
    const blitz::Array<T,2>& src, 
    const blitz::Array<bool,2>& src_mask, 
    blitz::Array<double,2>& dst, 
    blitz::Array<bool,2>& dst_mask,
    const double angle,
    const bob::ip::Rotate::Algorithm algo
)
{
  // Check input
  bob::core::array::assertZeroBase(src);
  bob::core::array::assertZeroBase(src_mask);
  bob::core::array::assertSameShape(src, src_mask);

  // Check output
  bob::core::array::assertZeroBase(dst);
  bob::core::array::assertZeroBase(dst_mask);
  bob::core::array::assertSameShape(dst, dst_mask);
  bob::core::array::assertSameShape(dst, bob::ip::getRotatedShape(src,angle));

  // Perform the rotation
  rotateNoCheck<T,true>(src, src_mask, dst, dst_mask, angle, algo);
}

/**
  * Rotate the given 3D (e.g. colored) input image to the given output image. 
  * The size of the output image must be identical to the size returned by bob::ip::getRotatedShape(src,angle).
  * @param src  The source image to rotate.
  * @param dst  The destination image that will hold the rotated image
  * @param angle  The angle in degrees the input image is rotated
  * @param algo  The algorithm used for image interpolation.
*/
template <typename T> 
  inline void bob::ip::rotate(
    const blitz::Array<T,3>& src, 
    blitz::Array<double,3>& dst,
    const double angle,
    const bob::ip::Rotate::Algorithm algo
)
{
  // Check input
  bob::core::array::assertZeroBase(src);

  // Check output
  bob::core::array::assertZeroBase(dst);
  bob::core::array::assertSameShape(dst, bob::ip::getRotatedShape(src,angle));

  // Perform the rotation
  for( int p=0; p<dst.extent(0); ++p) {
    // Prepare reference array to 2D slices
    const blitz::Array<T,2> src_slice = src( p, blitz::Range::all(), blitz::Range::all() );
    blitz::Array<double,2> dst_slice = dst( p, blitz::Range::all(), blitz::Range::all() );
    blitz::Array<bool,2> src_mask, dst_mask;
    rotateNoCheck<T,false>(src_slice, src_mask, dst_slice, dst_mask, angle, algo);
  }
}


/**
  * Rotate the given 3D (e.g. colored) input image to the given output image using the specified input mask.
  * The size of the output image and the output mask must be identical to the size returned by bob::ip::getRotatedShape(src,angle).
  * @note The input and output masks are only 2D!
  * @param src  The source image to rotate.
  * @param src_mask  The mask that should be applied
  * @param dst  The destination image that will hold the rotated image
  * @param dst_mask  The mask of the destination image
  * @param angle  The angle in degrees the input image is rotated
  * @param algo  The algorithm used for image interpolation.
*/
template <typename T> 
  inline void bob::ip::rotate(
    const blitz::Array<T,3>& src, 
    const blitz::Array<bool,2>& src_mask, 
    blitz::Array<double,3>& dst, 
    blitz::Array<bool,2>& dst_mask,
    const double angle,
    const bob::ip::Rotate::Algorithm algo
)
{
  // Check input
  bob::core::array::assertZeroBase(src);

  // Check output
  bob::core::array::assertZeroBase(dst);
  bob::core::array::assertSameShape(dst, bob::ip::getRotatedShape(src,angle));

  // Perform the rotation
  for( int p=0; p<dst.extent(0); ++p) {
    // Prepare reference array to 2D slices
    const blitz::Array<T,2> src_slice = src( p, blitz::Range::all(), blitz::Range::all() );
    blitz::Array<double,2> dst_slice = dst( p, blitz::Range::all(), blitz::Range::all() );
    rotateNoCheck<T,true>(src_slice, src_mask, dst_slice, dst_mask, angle, algo);
  }
}


/**
  * This function returns the shape of a rotated image, given an input 2D blitz array and an angle (in degrees). 
  * Please notice that the returned shape only depends on the shape of the image
  * and on the angle, but not on its content.
  * @param src The input 2D blitz array
  * @param angle The angle of the rotation (in degrees)
  * @return A TinyVector with the shape of the rotated image
  */
template<typename T>
  inline const blitz::TinyVector<int,2> bob::ip::getRotatedShape( 
    const blitz::Array<T,2>& src, 
    const double angle
) 
{
  // Initialize TinyVector
  blitz::TinyVector<int,2> dim;

  // Force the angle to be in range [-45,45]
  double angle_norm = angle;
  while(angle_norm < -45.)
    angle_norm += 360.;
  while(angle_norm > 315.)
    angle_norm -= 360.;

  // Determine the size of the rotated image
  if(angle_norm == 0. || angle_norm == 180.)
  {
    dim(0) = src.extent(0);
    dim(1) = src.extent(1);
  }
  else if(angle_norm == 90. || angle_norm == 270.)
  {
    dim(0) = src.extent(1);
    dim(1) = src.extent(0);
  }
  else {
    double rad_angle = angle_norm * M_PI / 180.;
    // Crop obtained sheared image
    const double dAbsSin = fabs(sin(rad_angle));
    const double dAbsCos = fabs(cos(rad_angle));
    dim(0) = floor(src.extent(0)*dAbsCos + src.extent(1)*dAbsSin + 0.5);
    dim(1) = floor(src.extent(0)*dAbsSin + src.extent(1)*dAbsCos + 0.5);
  }
  return dim;
}

/**
  * This function returns the shape of a rotated image, given an input 3D blitz array and an angle (in degrees). 
  * Please note that the returned shape only depends on the shape of the image
  * and of the angle, but not on its content.
  * @param src The input 3D blitz array
  * @param angle The angle of the rotation (in degrees)
  * @return A TinyVector with the shape of the rotated image
  */
template<typename T>
  inline const blitz::TinyVector<int,3> bob::ip::getRotatedShape( 
    const blitz::Array<T,3>& src, 
    const double angle
) 
{
  // Initialize TinyVector
  blitz::TinyVector<int,3> dim;
  dim(0) = src.extent(0); 

  // Call the getShapeRotated for the 2D case       
  blitz::Array<T,2> src_int = src(src.lbound(0), blitz::Range::all(), blitz::Range::all());
  const blitz::TinyVector<int,2> res_int = bob::ip::getRotatedShape(src_int, angle);
  dim(1) = res_int(0);
  dim(2) = res_int(1);

  return dim;
}


/**
 * Function to calculate the angle we need to rotate to level out two points horizontally.
 * @param left_y The vertical position of the left point
 * @param left_x The horizontal of the left point
 * @param right_y The vertical position of the right point
 * @param right_x The horizontal position of the right point
 * @return The angle (in degrees).
 */
inline double bob::ip::getAngleToHorizontal(
  const double left_y, 
  const double left_x,
  const double right_y, 
  const double right_x
)
{
  static const double RAD_TO_DEGREES   = 180. / M_PI;
  return std::atan2(right_y - left_y, right_x - left_x)
    * 
    RAD_TO_DEGREES;
}


#endif /* BOB_IP_ROTATE_H */
