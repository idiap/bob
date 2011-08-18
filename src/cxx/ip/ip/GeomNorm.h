/**
 * @file src/cxx/ip/ip/GeomNorm.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a class to perform geometric normalization of an
 * image. This means that the image is:
 *   1/ rotated with a given angle and a rotation center
 *   2/ rescaled according to a given scaling factor
 *   3/ cropped with respect to the point given and the additional
 *        cropping parameters.
 */

#ifndef TORCH5SPRO_IP_GEOM_NORM_H
#define TORCH5SPRO_IP_GROM_NORM_H

#include <boost/shared_ptr.hpp>
#include "core/array_assert.h"
#include "core/array_check.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/generateWithCenter.h"
#include "ip/Rotate.h"
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
    class GeomNorm
    {
      public:

        /**
          * @brief Constructor
          */
        GeomNorm(const double rotation_angle, const double scaling_factor, 
          const int crop_height, const int crop_width, 
          const int crop_offset_h, const int crop_offset_w);

        /**
          * @brief Destructor
          */
        virtual ~GeomNorm();

        /**
          * @brief Accessors
          */
        inline const double getRotationAngle() { return m_rotate->getAngle(); }
        inline const double getScalingFactor() { return m_scaling_factor; }
        inline const int getCropHeight() { return m_crop_height; }
        inline const int getCropWidth() { return m_crop_width; }
        inline const int getCropOffsetH() { return m_crop_offset_h; }
        inline const int getCropOffsetW() { return m_crop_offset_w; }

        /**
          * @brief Mutators
          */
        inline void setRotationAngle(const double angle) 
          { m_rotate->setAngle(angle); }
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
        template <typename T> 
        void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<double,2>& dst, const int rot_c_y, const int rot_c_x, 
          const int crop_ref_y, const int crop_ref_x);
        template <typename T> 
        void operator()(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
          blitz::Array<bool,2>& dst_mask, const int rot_c_y, const int rot_c_x,
          const int crop_ref_y, const int crop_ref_x);

      private:
        /**
          * @brief Process a 2D blitz Array/Image
          */
        template <typename T, bool mask>
        void processNoCheck(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
          blitz::Array<bool,2>& dst_mask, const int rot_c_y, const int rot_c_x,
          const int crop_ref_y, const int crop_ref_x);

        /**
          * Attributes
          */
        //double m_rotation_angle;
        boost::shared_ptr<Rotate> m_rotate;
        double m_scaling_factor;
        int m_crop_height;
        int m_crop_width;
        int m_crop_offset_h;
        int m_crop_offset_w;

        blitz::TinyVector<int,2> m_out_shape; 
        blitz::Array<double, 2> m_centered;
        blitz::Array<double, 2> m_rotated;
        blitz::Array<double, 2> m_scaled;

        blitz::Array<bool,2> m_mask_int1;
        blitz::Array<bool,2> m_mask_int2;
        blitz::Array<bool,2> m_mask_int3;
    };

    template <typename T> 
    void GeomNorm::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst, const int rot_c_y, const int rot_c_x,
      const int crop_ref_y, const int crop_ref_x) 
    { 
      // Check input
      tca::assertZeroBase(src);

      // Check output
      tca::assertZeroBase(dst);
      tca::assertSameShape(dst, m_out_shape);

      // Process
      blitz::Array<bool,2> src_mask, dst_mask;
      processNoCheck<T,false>( src, src_mask, dst, dst_mask, rot_c_y, rot_c_x,
        crop_ref_y, crop_ref_x);
    }

    template <typename T> 
    void GeomNorm::operator()(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
      blitz::Array<bool,2>& dst_mask, const int rot_c_y, const int rot_c_x,
      const int crop_ref_y, const int crop_ref_x) 
    { 
      // Check input
      tca::assertZeroBase(src);
      tca::assertZeroBase(src_mask);
      tca::assertSameShape(src,src_mask);

      // Check output
      tca::assertZeroBase(dst);
      tca::assertZeroBase(dst_mask);
      tca::assertSameShape(dst, dst_mask);
      tca::assertSameShape(dst, m_out_shape);

      // Process
      processNoCheck<T,true>( src, src_mask, dst, dst_mask, rot_c_y, rot_c_x,
        crop_ref_y, crop_ref_x);
    }

    // TODO: Refactor with Geometry module to keep track of the cropping
    // point coordinates after each operation
    template <typename T, bool mask> 
    void GeomNorm::processNoCheck(const blitz::Array<T,2>& src,
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
      blitz::Array<bool,2>& dst_mask, const int rot_c_y, const int rot_c_x,
      const int crop_ref_y, const int crop_ref_x) 
    { 
      // 0/ Cast input to double
      blitz::Array<double,2> src_d = Torch::core::cast<double>(src);
 
      // 1/ Expand the image such the the point (yc,xc) is at the center
      blitz::TinyVector<int,2> shape = 
        getGenerateWithCenterShape(src_d, rot_c_y, rot_c_x);
      blitz::TinyVector<int,2> offset = 
        getGenerateWithCenterOffset(src_d, rot_c_y, rot_c_x);
      if( !tca::hasSameShape(m_centered, shape) ) {
        m_centered.resize( shape );
        if(mask)
          m_mask_int1.resize(shape);
      }
      if(mask)
        generateWithCenter(src_d, src_mask, m_centered, m_mask_int1,
          rot_c_y, rot_c_x);
      else
        generateWithCenter(src_d, m_centered, rot_c_y, rot_c_x);

      // new coordinate of the cropping reference point
      double crop_ref_y1 = offset(0) + crop_ref_y;
      double crop_ref_x1 = offset(1) + crop_ref_x;

      // 2/ Rotate to align the image with the x-axis
      shape = m_rotate->getOutputShape(m_centered, m_rotate->getAngle());
      if( !tca::hasSameShape(m_rotated, shape) ) {
        m_rotated.resize( shape );
        if(mask)
          m_mask_int2.resize(shape);
      }
      if(mask)
        m_rotate->operator()(m_centered, m_mask_int1, m_rotated, m_mask_int2);
      else
        m_rotate->operator()(m_centered, m_rotated);

      // new coordinate of the cropping reference point
      crop_ref_y1 = crop_ref_y1 - (m_centered.extent(0)-1)/2.;
      crop_ref_x1 = crop_ref_x1 - (m_centered.extent(1)-1)/2.;
      double crop_ref_y2 = crop_ref_y1 * cos(m_rotate->getAngle()) - crop_ref_x1 * sin(m_rotate->getAngle()) + (m_rotated.extent(0)-1)/2.;
      double crop_ref_x2 = crop_ref_x1 * cos(m_rotate->getAngle()) + crop_ref_y1 * sin(m_rotate->getAngle()) + (m_rotated.extent(1)-1)/2.;

      // 3/ Scale with the given scaling factor
      shape(0) = static_cast<int>(floor(m_rotated.extent(0) * m_scaling_factor + 0.5));
      shape(1) = static_cast<int>(floor(m_rotated.extent(1) * m_scaling_factor + 0.5));
      if( !tca::hasSameShape(m_scaled, shape) ) {
        m_scaled.resize( shape );
        if(mask)
          m_mask_int3.resize(shape);
      }
      if(mask)
        scale(m_rotated, m_mask_int2, m_scaled, m_mask_int3);
      else
        scale(m_rotated, m_scaled);

      // new coordinate of the cropping reference point
      int crop_ref_y3 = static_cast<int>(floor(crop_ref_y2 * m_scaling_factor + 0.5));
      int crop_ref_x3 = static_cast<int>(floor(crop_ref_x2 * m_scaling_factor + 0.5));

      // 4/ Crop the face
      if(mask)
        crop(m_scaled, m_mask_int3, dst, dst_mask, crop_ref_y3-m_crop_offset_h,
          crop_ref_x3-m_crop_offset_w, m_crop_height, m_crop_width, true, true);
      else
        crop(m_scaled, dst, crop_ref_y3-m_crop_offset_h, 
          crop_ref_x3-m_crop_offset_w, m_crop_height, m_crop_width, true, true);
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_IP_GEOM_NORM_H */
