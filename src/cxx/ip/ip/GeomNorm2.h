/**
 * @file src/cxx/ip/ip/geomNorm2.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a class to perform geometric normalization of an
 * image. This means that the image is:
 *   1/ rotated with a given angle and a rotation center
 *   2/ rescaled according to a given scaling factor
 *   3/ cropped with respect to the point given and the additional
 *        cropping parameters.
 */

#ifndef TORCH5SPRO_IP_GEOM_NORM2_H
#define TORCH5SPRO_IP_GROM_NORM2_H

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
     * an image. This means that the image is:
     *   1/ rotated with a given angle and a rotation center
     *   2/ rescaled according to a given scaling factor
     *   3/ cropped with respect to the point given and the additional
     *        cropping parameters.
     */
    class GeomNormNew
    {
      public:

        /**
          * @brief Constructor
          */
        GeomNormNew(const double rotation_angle, const int scaling_factor, 
          const int crop_height, const int crop_width, 
          const int crop_offset_h, const int crop_offset_w);

        /**
          * @brief Destructor
          */
        virtual ~GeomNormNew();

        /**
          * @brief Accessors
          */
        inline const double getRotationAngle() { return m_rotation_angle; }
        inline const double getScalingFactor() { return m_scaling_factor; }
        inline const int getCropHeight() { return m_crop_height; }
        inline const int getCropWidth() { return m_crop_width; }
        inline const int getCropOffsetH() { return m_crop_offset_h; }
        inline const int getCropOffsetW() { return m_crop_offset_w; }

        /**
          * @brief Mutators
          */
        inline void setRotationAngle(const double angle) 
          { m_rotation_angle = angle; }
        inline void setScalingFactor(const double scaling_factor) 
          { m_scaling_factor = scaling_factor; }
        inline void setCropHeight(const int crop_h) 
          { m_crop_height = crop_h; }
        inline void setCropWidth(const int crop_w) 
          { m_crop_width = crop_w; }
        inline void setCropOffsetH(const int crop_dh) 
          { m_crop_offset_h = crop_dh; }
        inline void setCropOffsetW(const int crop_dw) 
          { m_crop_offset_w = crop_dw; }

        /**
          * @brief Process a 2D blitz Array/Image by applying the geometric
          * normalization
          */
        template <typename T> void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<double,2>& dst, const int rot_c_y, const int rot_c_x, 
          const int crop_ref_y, const int crop_ref_x);

      private:
        /**
          * Attributes
          */
        double m_rotation_angle;
        double m_scaling_factor;
        int m_crop_height;
        int m_crop_width;
        int m_crop_offset_h;
        int m_crop_offset_w;

        blitz::TinyVector<int,2> m_out_shape; 
        blitz::Array<double, 2> m_centered;
        blitz::Array<double, 2> m_rotated;
        blitz::Array<double, 2> m_scaled;
    };

    // TODO: Refactor with Geometry module to keep track of the cropping
    // point coordinates after each operation
    template <typename T> 
    void GeomNormNew::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst, const int rot_c_y, const int rot_c_x,
      const int crop_ref_y, const int crop_ref_x) 
    { 
      // Check input
      tca::assertZeroBase(src);

      // Check output
      tca::assertZeroBase(dst);
      tca::assertSameShape(dst, m_out_shape);

      // 0/ Cast input to double
      blitz::Array<double,2> src_d = Torch::core::cast<double>(src);
 
      // 1/ Expand the image such the the point (yc,xc) is at the center
      blitz::TinyVector<int,2> shape = 
        getGenerateWithCenterShape(src_d, rot_c_y, rot_c_x);
      blitz::TinyVector<int,2> offset = 
        getGenerateWithCenterOffset(src_d, rot_c_y, rot_c_x);
      if( !tca::hasSameShape(m_centered, shape) )
        m_centered.resize( shape );
      generateWithCenter(src_d, m_centered, rot_c_y, rot_c_x);

      // new coordinate of the cropping reference point
      int crop_ref_y1 = offset(0) + crop_ref_y;
      int crop_ref_x1 = offset(1) + crop_ref_x;

      // 2/ Rotate to align the image with the x-axis
      shape = getShapeRotated(m_centered, m_rotation_angle);
      if( !tca::hasSameShape(m_rotated, shape) )
        m_rotated.resize( shape );
      rotate(m_centered, m_rotated, m_rotation_angle);

      // new coordinate of the cropping reference point
      crop_ref_y1 = crop_ref_y1 - m_centered.extent(0)/2;
      crop_ref_x1 = crop_ref_x1 - m_centered.extent(1)/2;
      int crop_ref_y2 = crop_ref_y1 * cos(m_rotation_angle) - crop_ref_x1 * sin(m_rotation_angle) + m_rotated.extent(0)/2;
      int crop_ref_x2 = crop_ref_x1 * cos(m_rotation_angle) + crop_ref_y1 * sin(m_rotation_angle) + m_rotated.extent(1)/2;

      // 3/ Scale with the given scaling factor
      shape(0) = m_rotated.extent(0) * m_scaling_factor;
      shape(1) = m_rotated.extent(1) * m_scaling_factor;
      if( !tca::hasSameShape(m_scaled, shape) )
        m_scaled.resize( shape );
      scale(m_rotated, m_scaled);

      // new coordinate of the cropping reference point
      int crop_ref_y3 = crop_ref_y2 * m_scaling_factor;
      int crop_ref_x3 = crop_ref_x2 * m_scaling_factor;

      // 4/ Crop the face
      crop(m_scaled, dst, crop_ref_y3 - m_crop_offset_h, 
        crop_ref_x3 - m_crop_offset_w, m_crop_height, m_crop_width);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_GEOM_NORM2_H */
