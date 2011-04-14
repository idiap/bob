/**
 * @file src/cxx/ip/ip/geomNorm.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a class to perform geometric normalization of an
 * image. This means that given two points, the image is:
 *   1/ rotated such that the angle between the x-axis and the line connecting
 *       the two points is set to some given value,
 *   2/ rescaled such that the distance between the two (rotated) points is set
 *       to some given value,
 *   3/ cropped with respect to the two points given, and to a given size.
 */

#ifndef TORCH5SPRO_IP_GEOM_NORM_H
#define TORCH5SPRO_IP_GROM_NORM_H

#include "core/array_assert.h"
#include "core/array_check.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/shiftToCenter.h"
#include "ip/rotate.h"
#include "ip/scale.h"
#include "ip/crop.h"

namespace tca = Torch::core::array;

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
     * @brief This file defines a class to perform geometric normalization of 
     * an image. This means that given two points, the image is:
     *   1/ rotated such that the angle between the x-axis and the line
     *       connecting the two points is set to some given value,
     *   2/ rescaled such that the distance between the two (rotated) points
     *       is set to some given value,
     *   3/ cropped with respect to the two points given, and to a given size.
     * In particular, this is the case when we want to crop a face from an 
     * image, into a rotated sample image of a given size.
    */
    class GeomNorm
    {
      public:

        /**
          * @brief Constructor
          */
        GeomNorm(const int eyes_distance, const int height, const int width, 
          const int border_h, const int border_w);

        /**
          * @brief Destructor
          */
        virtual ~GeomNorm();

        /**
          * @brief Process a 2D blitz Array/Image by applying the geometric
          * normalization
          */
        template <typename T> void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<double,2>& dst, const int y1, const int x1, 
          const int y2, const int x2);

      private:
        /**
          * Attributes
          */
        int m_eyes_distance;
        int m_height;
        int m_width;
        int m_border_h;
        int m_border_w;

        blitz::TinyVector<int,2> m_out_shape; 
        blitz::Array<double, 2> m_src_d;
        blitz::Array<double, 2> m_centered;
        blitz::Array<double, 2> m_rotated;
        blitz::Array<double, 2> m_scaled;
    };

    template <typename T> 
    void GeomNorm::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst, const int y1, const int x1,
      const int y2, const int x2) 
    { 
      // Check input
      tca::assertZeroBase(src);

      // Check output
      tca::assertZeroBase(dst);
      tca::assertSameShape(dst,m_out_shape);

      // Compute the coordinates of the point defined as the center of the
      // segment of the two points (/eyes).
      const int yc = abs(y1+y2) / 2;
      const int xc = abs(x1+x2) / 2;
    
      // 0/ Cast input to double
      blitz::Array<double,2> src_d = Torch::core::cast<double>(src);
 
      // 1/ Expand the image such the the point (yc,xc) is at the center
      blitz::TinyVector<int,2> shape = 
        getGenerateWithCenterShape(src_d,yc,xc);
      if( !tca::hasSameShape(m_centered, shape) )
        m_centered.resize( shape );
      generateWithCenter(src_d,m_centered,yc,xc);
        
      // 2/ Rotate to align the image with the x-axis
      const double angle = getAngleToHorizontal(y1, x1, y2, x2);
      shape = getShapeRotated(m_centered, angle);
      if( !tca::hasSameShape(m_rotated, shape) )
        m_rotated.resize( shape );
      rotate(m_centered,m_rotated,angle);

      // 3/ Rescale such that the distance between the points/eyes is as expected
      const double eyes_distance_init = 
        sqrt((y1-y2)*(y1-y2) + (x1-x2)*(x1-x2));

      const double scale_factor = m_eyes_distance / eyes_distance_init;
      shape(0) = src.extent(0) * scale_factor;
      shape(1) = src.extent(1) * scale_factor;
      if( !tca::hasSameShape(m_scaled, shape) )
        m_scaled.resize( shape );
      scale(m_rotated,m_scaled);

      // 4/ Crop the face
      cropFace(m_scaled, dst, m_eyes_distance, m_border_h, m_border_w);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_GEOM_NORM_H */
