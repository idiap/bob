/**
 * @file src/cxx/ip/ip/Rotate.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to rotate a 2D or 3D array/image.
 * The shearing-based algorithm is strongly inspired by the following article:
 * 'A Fast Algorithm for General Raster Rotation', Alan Paeth, in the 
 * proceedings of Graphics Interface '86, p. 77-81.
 * The notes of Tobin Fricke about this article might also be of interest.
 */

#ifndef TORCH5SPRO_IP_ROTATE_H
#define TORCH5SPRO_IP_ROTATE_H

#include <blitz/array.h>
#include "core/array_assert.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/shear.h"
#include "ip/crop.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {
    namespace detail {
    }

    class Rotate
    {
      public:
        /**
          * @brief Internal enumeration of the possible algorithms
          */
        typedef enum Algorithm {
          Shearing,
          BilinearInterp
        } Algorithm;

        /**
          * @brief Constructor
          */
        Rotate(const double angle, const Algorithm algo=Shearing);

        /**
          * @brief Destructor
          */
        virtual ~Rotate();

        /**
          * Accessors
          */
        double getAngle() const { return m_angle; }
        Algorithm getAlgorithm() const { return m_algo; }

        /**
          * Mutators
          */
        void setAngle(const double angle) { m_angle = angle; }
        void setAlgorithm(Algorithm algo) 
          { m_algo = algo; }

        /**
          * @brief Process a 2D blitz Array/Image by applying the rotation
          */
        template <typename T> void operator()(const blitz::Array<T,2>& src,
          blitz::Array<double,2>& dst);
        template <typename T> void operator()(const blitz::Array<T,2>& src,
          blitz::Array<double,2>& dst, const double angle);
        template <typename T> void operator()(const blitz::Array<T,2>& src,
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask);
        template <typename T> void operator()(const blitz::Array<T,2>& src,
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
          blitz::Array<bool,2>& dst_mask, const double angle);
      
        /**
          * @brief Process a 3D blitz Array/Image by applying the rotation
          */
        template <typename T> void operator()(const blitz::Array<T,3>& src,
          blitz::Array<double,3>& dst);
        template <typename T> void operator()(const blitz::Array<T,3>& src,
          blitz::Array<double,3>& dst, const double angle);
      
        /**
          * @brief Returns the shape of the output, for a given src 
          *    array/image and angle.
          */
        template<typename T>
        static const blitz::TinyVector<int,2> getOutputShape( 
          const blitz::Array<T,2>& src, const double angle);
        template<typename T>
        static const blitz::TinyVector<int,3> getOutputShape( 
          const blitz::Array<T,3>& src, const double angle);
        
  
      private:

        /**
          * @brief Private methods used to rotate an array with a 'common'
          *   angle (0, 90, 180 and 270 degrees)
          */
        template<typename T, bool mask>
        static void rotateNoCheck_0(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask);
        template<typename T, bool mask>
        static void rotateNoCheck_90(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask);
        template<typename T, bool mask>
        static void rotateNoCheck_180(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask);
        template<typename T, bool mask>
        static void rotateNoCheck_270(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask);
        template<typename T, bool mask>

        /**
          * @brief Private method which selects the rotating algorithm
          */
        void rotateNoCheck(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask);

        /**
          * @brief Private method which applies the shearing-based rotating 
          *   algorithm
          */
        template<typename T, bool mask>
        void rotateShearingNoCheck(const blitz::Array<T,2>& src,
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
          blitz::Array<bool,2>& dst_mask, const double angle);

        /**
          * Attributes
          */
        double m_angle;
        Algorithm m_algo;

        int m_previous_height;
        int m_previous_width;
        blitz::Array<double,2> m_dst_int1;
        blitz::Array<double,2> m_dst_int2;
        blitz::Array<double,2> m_dst_int3;
        blitz::Array<double,2> m_dst_int4;
        blitz::Array<bool,2> m_mask_int1;
        blitz::Array<bool,2> m_mask_int2;
        blitz::Array<bool,2> m_mask_int3;
        blitz::Array<bool,2> m_mask_int4;
    };

    template <typename T> 
    void Rotate::operator()(const blitz::Array<T,2>& src,
      blitz::Array<double,2>& dst, const double angle)
    {
      setAngle(angle);
      this->operator()(src,dst);
    }

    template <typename T> 
    void Rotate::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst)
    {
      // Check input
      Torch::core::array::assertZeroBase(src);

      // Check output
      Torch::core::array::assertZeroBase(dst);
      Torch::core::array::assertSameShape(dst, getOutputShape(src,m_angle));

      // Perform the rotation
      blitz::Array<bool,2> src_mask, dst_mask;
      rotateNoCheck<T,false>(src, src_mask, dst, dst_mask);
    }

    template <typename T> 
    void Rotate::operator()(const blitz::Array<T,2>& src,
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
      blitz::Array<bool,2>& dst_mask, const double angle)
    {
      setAngle(angle);
      this->operator()(src, src_mask, dst, dst_mask);
    }

    template <typename T> 
    void Rotate::operator()(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
      blitz::Array<bool,2>& dst_mask)
    {
      // Check input
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(src_mask);
      Torch::core::array::assertSameShape(src, src_mask);

      // Check output
      Torch::core::array::assertZeroBase(dst);
      Torch::core::array::assertZeroBase(dst_mask);
      Torch::core::array::assertSameShape(dst, dst_mask);
      Torch::core::array::assertSameShape(dst, getOutputShape(src,m_angle));

      // Perform the rotation
      rotateNoCheck<T,true>(src, src_mask, dst, dst_mask);
    }


    template <typename T> 
    void Rotate::operator()(const blitz::Array<T,3>& src,
      blitz::Array<double,3>& dst, const double angle)
    {
      setAngle(angle);
      this->operator()(src,dst);
    }

    template <typename T> 
    void Rotate::operator()(const blitz::Array<T,3>& src, 
      blitz::Array<double,3>& dst)
    {
      // Check input
      Torch::core::array::assertZeroBase(src);

      // Check output
      Torch::core::array::assertZeroBase(dst);
      Torch::core::array::assertSameShape(dst, getOutputShape(src,m_angle));

      // Perform the rotation
      for( int p=0; p<dst.extent(0); ++p) {
        // Prepare reference array to 2D slices
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<double,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<bool,2> src_mask, dst_mask;
        rotateNoCheck<T,false>(src_slice, src_mask, dst_slice, dst_mask);
      }
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image of a given type
      *   with an angle of 0 degrees.
      * @warning No check is performed on the dst blitz::array/image.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T, bool mask>
    void Rotate::rotateNoCheck_0(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask)
    { 
      for( int y=0; y<dst.extent(0); ++y)
        for( int x=0; x<dst.extent(1); ++x)
          dst(y,x) = Torch::core::cast<double>(src(y,x));
      if(mask) {
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst_mask(y,x) = Torch::core::cast<double>(src_mask(y,x));
      }
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image of a given type
      *   with an angle of 90 degrees.
      * @warning No check is performed on the dst blitz::array/image.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T, bool mask>
    void Rotate::rotateNoCheck_90(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask)
    { 
      for( int y=0; y<dst.extent(0); ++y)
        for( int x=0; x<dst.extent(1); ++x)
          dst(y,x) = static_cast<double>(src( x, (src.extent(1)-1-y) ));
      if(mask) {
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst_mask(y,x) = Torch::core::cast<double>(
                              src_mask(x, (src.extent(1)-1-y) ));
      }
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image of a given type
      *   with an angle of 180 degrees.
      * @warning No check is performed on the dst blitz::array/image.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T, bool mask>
    void Rotate::rotateNoCheck_180(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask)
    { 
      for( int y=0; y<dst.extent(0); ++y)
        for( int x=0; x<dst.extent(1); ++x)
          dst(y,x) = Torch::core::cast<double>(src( (src.extent(0)-1-y),
                                              (src.extent(1)-1-x) ));
      if(mask) {
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst_mask(y,x) = Torch::core::cast<double>(
                              src_mask( (src.extent(0)-1-y), 
                                        (src.extent(1)-1-x) ));
      }
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image of a given type
      *   with an angle of 270 degrees.
      * @warning No check is performed on the dst blitz::array/image.
      * @param src The input blitz array
      * @param dst The output blitz array
      */
    template<typename T, bool mask>
    void Rotate::rotateNoCheck_270(const blitz::Array<T,2>& src,
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask)
    { 
      for( int y=0; y<dst.extent(0); ++y)
        for( int x=0; x<dst.extent(1); ++x)
          dst(y,x) = Torch::core::cast<double>(src( (src.extent(0)-1-x), y ));
      if(mask) {
        for( int y=0; y<dst.extent(0); ++y)
          for( int x=0; x<dst.extent(1); ++x)
            dst_mask(y,x) = Torch::core::cast<double>(
                              src_mask( (src.extent(0)-1-x), y ));
      }
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image 
      * @warning No check is performed on the dst blitz::array/image.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param angle The angle of the rotation (in degrees)
      * @param alg The algorithm which should be used to perform the 
      *   rotation.
      */
    template<typename T, bool mask>
    void Rotate::rotateNoCheck(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask)
    { 
      // Force the angle to be in range [-45,315] 
      double angle_norm = m_angle;
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

      switch(m_algo)
      {
        case Rotate::Shearing:
          rotateShearingNoCheck<T,mask>(src, src_mask, dst, dst_mask, 
            angle_norm);
          break;
        default:
          throw Torch::ip::UnknownRotatingAlgorithm();
      }
    }

    /**
      * @brief Function which rotates a 2D blitz::array/image using a
      *   shearing algorithm.
      * @warning No check is performed on the dst blitz::array/image.
      * @param src The input blitz array
      * @param dst The output blitz array
      * @param angle The angle of the rotation (in degrees)
      * @param alg The algorithm which should be used to perform the 
      *   rotation.
      */
    template<typename T, bool mask>
    void Rotate::rotateShearingNoCheck(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask, const double angle)
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

      // Declare an intermediate arrays
      // Perform simple rotation. After that, there is one more
      //  rotation to do with an angle in [-45,45].
      if( quadrant == 0 ) {
        if( m_dst_int1.extent(0) != src.extent(0) || 
            m_dst_int1.extent(1) != src.extent(1) ) {
          m_dst_int1.resize( src.extent(0), src.extent(1) );
          if(mask)
            m_mask_int1.resize( src.extent(0), src.extent(1) );
        }
        rotateNoCheck_0<T,mask>(src, src_mask, m_dst_int1, m_mask_int1);
      }
      else if( quadrant == 1) {
        if( m_dst_int1.extent(0) != src.extent(1) || 
            m_dst_int1.extent(1) != src.extent(0) ) {
          m_dst_int1.resize( src.extent(1), src.extent(0) );
          if(mask)
            m_mask_int1.resize( src.extent(1), src.extent(0) );
        }
        rotateNoCheck_90<T,mask>(src, src_mask, m_dst_int1, m_mask_int1);
      }
      else if( quadrant == 2) {
        if( m_dst_int1.extent(0) != src.extent(0) || 
            m_dst_int1.extent(1) != src.extent(1) ) {
          m_dst_int1.resize( src.extent(0), src.extent(1) );
          if(mask)
            m_mask_int1.resize( src.extent(0), src.extent(1) );
        }
        rotateNoCheck_180<T,mask>(src, src_mask, m_dst_int1, m_mask_int1);
      }
      else { // quadrant == 3
        if( m_dst_int1.extent(0) != src.extent(1) || 
            m_dst_int1.extent(1) != src.extent(0) ) {
          m_dst_int1.resize( src.extent(1), src.extent(0) );
          if(mask)
            m_mask_int1.resize( src.extent(1), src.extent(0) );
        }
        rotateNoCheck_270<T,mask>(src, src_mask, m_dst_int1, m_mask_int1);
      }

      // Compute shearing values required for the rotation
      const double shear_x = -tan( rad_angle / 2. );
      const double shear_y = sin( rad_angle );

      // Performs first shear (shearX)
      const blitz::TinyVector<int,2> s1 = 
        getShearXShape(m_dst_int1, shear_x);
      if( m_dst_int2.extent(0) != s1(0) || 
          m_dst_int2.extent(1) != s1(1) ) {
        m_dst_int2.resize(s1);
        if(mask)
          m_mask_int2.resize( s1 );
      }
      if(mask)
        shearX( m_dst_int1, m_mask_int1, m_dst_int2, m_mask_int2, 
            shear_x, true);
      else 
        shearX( m_dst_int1, m_dst_int2, shear_x, true);
      // Performs second shear (shearY)
      const blitz::TinyVector<int,2> s2 = 
        getShearYShape(m_dst_int2, shear_y);
      if( m_dst_int3.extent(0) != s2(0) || 
          m_dst_int3.extent(1) != s2(1) ) {
        m_dst_int3.resize(s2);
        if(mask)
          m_mask_int3.resize( s2 );
      }
      if(mask)
        shearY( m_dst_int2, m_mask_int2, m_dst_int3, m_mask_int3,
            shear_y, true);
      else
        shearY( m_dst_int2, m_dst_int3, shear_y, true);
      // Performs third shear (shearX)
      const blitz::TinyVector<int,2> s3 = 
        getShearXShape(m_dst_int3, shear_x);
      if( m_dst_int4.extent(0) != s3(0) || 
          m_dst_int4.extent(1) != s3(1) ) {
        m_dst_int4.resize(s3);
        if(mask)
          m_mask_int4.resize( s3 );
      }
      if(mask)
        shearX( m_dst_int3, m_mask_int3, m_dst_int4, m_mask_int4,
            shear_x, true);
      else
        shearX( m_dst_int3, m_dst_int4, shear_x, true);

      // Crop obtained sheared image
      const blitz::TinyVector<int,2> crop_d = getOutputShape(src, angle);

      int crop_x = (m_dst_int4.extent(1) - crop_d(1)) / 2;
      int crop_y = (m_dst_int4.extent(0) - crop_d(0)) / 2;
      if(mask)
        crop( m_dst_int4, m_mask_int4, dst, dst_mask, crop_y, crop_x,
            crop_d(0), crop_d(1), true, true);
      else 
        crop( m_dst_int4, dst, crop_y, crop_x, crop_d(0), crop_d(1),
            true, true);
    }



    /**
      * @brief Function which returns the shape of a rotated image, given
      *   an input 2D blitz array and an angle (in degrees). Please notice
      *   that the returned shape only depends on the shape of the image
      *   and of the angle, but not on its content.
      * @param src The input 2D blitz array
      * @param angle The angle of the rotation (in degrees)
      * @return A TinyVector with the shape of the rotated image
      */
    template<typename T>
    const blitz::TinyVector<int,2> Rotate::getOutputShape( 
      const blitz::Array<T,2>& src, const double angle) 
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
      * @brief Function which returns the shape of a rotated image, given
      *   an input 3D blitz array and an angle (in degrees). Please notice
      *   that the returned shape only depends on the shape of the image
      *   and of the angle, but not on its content.
      * @param src The input 3D blitz array
      * @param angle The angle of the rotation (in degrees)
      * @return A TinyVector with the shape of the rotated image
      */
    template<typename T>
    const blitz::TinyVector<int,3> Rotate::getOutputShape( 
      const blitz::Array<T,3>& src, const double angle) 
    {
      // Initialize TinyVector
      blitz::TinyVector<int,3> dim;
      dim(0) = src.extent(0); 

      // Call the getShapeRotated for the 2D case       
      blitz::Array<T,2> src_int = src(src.lbound(0), blitz::Range::all(), 
        blitz::Range::all());
      const blitz::TinyVector<int,2> res_int =
        Rotate::getOutputShape(src_int, angle);
      dim(1) = res_int(0);
      dim(2) = res_int(1);

      return dim;
    }

    /**
     * @brief Function to calculate the angle we need to rotate to level out two points
     *  horizontally
     * @param left_h The height of the left point
     * @param left_w The width of the left point
     * @param right_h The height of the right point
     * @param right_w The width of the right point
     */
    inline double getAngleToHorizontal(const int left_h, const int left_w,
					      const int right_h, const int right_w)
    {
	    static const double RAD_TO_DEGREES   = 180. / M_PI;
	    return std::tan(1.0 * (right_h - left_h) / (right_w - left_w)) 
		    * 
		    RAD_TO_DEGREES;
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_ROTATE_H */
