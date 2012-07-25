/**
 * @file cxx/ip/ip/FaceEyesNorm.h
 * @date Thu Apr 14 21:03:45 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#ifndef BOB_IP_FACE_EYES_NORM_H
#define BOB_IP_FACE_EYES_NORM_H

#include <boost/shared_ptr.hpp>
#include "core/array_assert.h"
#include "core/array_check.h"
#include "ip/GeomNorm.h"
#include "ip/rotate.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
     * @brief A class to perform a geometric normalization of a face based 
     * on the eye center locations.
     */
    class FaceEyesNorm
    {
      public:

        /**
          * @brief Constructor
          */
        FaceEyesNorm( const int eyes_distance, const int crop_height, 
          const int crop_width, const int crop_offset_h, 
          const int crop_offset_w);

        /**
         * @brief Copy constructor
         */
        FaceEyesNorm(const FaceEyesNorm& other); 

        /**
          * @brief Destructor
          */
        virtual ~FaceEyesNorm() {}

        /**
         * @brief Assignment operator
         */
        FaceEyesNorm& operator=(const FaceEyesNorm& other);

        /**
         * @brief Equal to
         */
        bool operator==(const FaceEyesNorm& b) const;
        /**
         * @brief Not equal to
         */
        bool operator!=(const FaceEyesNorm& b) const; 
 
        /**
          * @brief Accessors
          */
        int getEyesDistance() const { return m_eyes_distance; }
        int getCropHeight() const { return m_crop_height; }
        int getCropWidth() const { return m_crop_width; }
        int getCropOffsetH() const { return m_crop_offset_h; }
        int getCropOffsetW() const { return m_crop_offset_w; }

        /**
          * @brief Mutators
          */
        void setEyesDistance(const int eyes_distance) 
          { m_eyes_distance = eyes_distance; }
        void setCropHeight(const int crop_h) 
          { m_crop_height = crop_h; m_geom_norm->setCropHeight(crop_h); }
        void setCropWidth(const int crop_w) 
          { m_crop_width = crop_w; m_geom_norm->setCropWidth(crop_w); }
        void setCropOffsetH(const int crop_dh) 
          { m_crop_offset_h = crop_dh; m_geom_norm->setCropOffsetH(crop_dh); }
        void setCropOffsetW(const int crop_dw) 
          { m_crop_offset_w = crop_dw; m_geom_norm->setCropOffsetW(crop_dw); }

        /**
          * @brief Process a 2D face image by applying the geometric
          * normalization
          */
        template <typename T> void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<double,2>& dst, const int e1_y, const int e1_x, 
          const int e2_y, const int e2_x) const;
        template <typename T> void operator()(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
          blitz::Array<bool,2>& dst_mask, const int e1_y, const int e1_x, 
          const int e2_y, const int e2_x) const;

      private:
        template <typename T, bool mask> 
        void processNoCheck(const blitz::Array<T,2>& src, 
          const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst, 
          blitz::Array<bool,2>& dst_mask, const int e1_y, const int e1_x, 
          const int e2_y, const int e2_x) const;

        /**
          * Attributes
          */
        double m_eyes_distance;
        int m_crop_height;
        int m_crop_width;
        int m_crop_offset_h;
        int m_crop_offset_w;

        blitz::TinyVector<int,2> m_out_shape;
        boost::shared_ptr<GeomNorm> m_geom_norm;
    };

    template <typename T> 
    inline void FaceEyesNorm::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<double,2>& dst, const int e1_y, const int e1_x,
      const int e2_y, const int e2_x) 
    { 
      // Check input
      bob::core::array::assertZeroBase(src);

      // Check output
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertSameShape(dst, m_out_shape);

      // Process
      blitz::Array<bool,2> src_mask, dst_mask;
      processNoCheck<T,false>(src, src_mask, dst, dst_mask, e1_y, e1_x, e2_y, 
        e2_x); 
    }

    template <typename T> 
    inline void FaceEyesNorm::operator()(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask, const int e1_y, const int e1_x,
      const int e2_y, const int e2_x) 
    { 
      // Check input
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(src_mask);
      bob::core::array::assertSameShape(src,src_mask);

      // Check output
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertZeroBase(dst_mask);
      bob::core::array::assertSameShape(dst,dst_mask);
      bob::core::array::assertSameShape(dst, m_out_shape);

      // Process
      processNoCheck<T,true>(src, src_mask, dst, dst_mask, e1_y, e1_x, e2_y, 
        e2_x); 
    }

    template <typename T, bool mask> 
    inline void FaceEyesNorm::processNoCheck(const blitz::Array<T,2>& src, 
      const blitz::Array<bool,2>& src_mask, blitz::Array<double,2>& dst,
      blitz::Array<bool,2>& dst_mask, const int e1_y, const int e1_x,
      const int e2_y, const int e2_x) 
    { 
      // Get angle to horizontal
      const double angle = getAngleToHorizontal(e1_y, e1_x, e2_y, e2_x);
      m_geom_norm->setRotationAngle( angle);

      // Get scaling factor
      const double eyes_distance = sqrt( (e1_y-e2_y)*(e1_y-e2_y) + (e1_x-e2_x)*(e1_x-e2_x) );
      m_geom_norm->setScalingFactor( m_eyes_distance / eyes_distance);

      // Get the center (of the eye centers segment)
      int center_y = (e1_y + e2_y) / 2;
      int center_x = (e1_x + e2_x) / 2;

      // Perform the normalization
      if(mask)
        m_geom_norm->operator()(src, src_mask, dst, dst_mask, center_y, center_x);
      else
        m_geom_norm->operator()(src, dst, center_y, center_x);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_IP_FACE_EYES_NORM_H */
