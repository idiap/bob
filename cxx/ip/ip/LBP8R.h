/**
 * @file cxx/ip/ip/LBP8R.h
 * @date Wed Apr 20 21:44:36 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to compute the LBP8R.
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

#ifndef BOB5SPRO_IP_LBP8R_H
#define BOB5SPRO_IP_LBP8R_H

#include <blitz/array.h>
#include <algorithm>
#include <cmath>
#include "core/array_assert.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/LBP.h"
#include "sp/interpolate.h"

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
      * @brief This class allows to extract Local Binary Pattern-like features
      *   based on 8 neighbour pixels.
      *   For more information, please refer to the following article:
      *     "Face Recognition with Local Binary Patterns", from T. Ahonen,
      *     A. Hadid and M. Pietikainen
      *     in the proceedings of the European Conference on Computer Vision
      *     (ECCV'2004), p. 469-481
      */
    class LBP8R: public LBP
    {
      public:
        /**
          * @brief Constructor
          */
        LBP8R(const double R=1., const bool circular=false, 
            const bool to_average=false, const bool add_average_bit=false, 
            const bool uniform=false, const bool rotation_invariant=false, const int eLBP_type=0);

        /**
          * @brief Destructor
          */
        virtual ~LBP8R() { }

		    /**
          * @brief Return the maximum number of labels for the current LBP 
          *   variant
          */
  		  virtual int getMaxLabel() const;

        /**
          * @brief Extract LBP features from a 2D blitz::Array, and save 
          *   the resulting LBP codes in the dst 2D blitz::Array.
          */
        template <typename T> 
        void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<uint16_t,2>& dst) const;
        void 
        operator()(const blitz::Array<uint8_t,2>& src, 
            blitz::Array<uint16_t,2>& dst) const 
          { operator()<uint8_t>(src, dst); }
        void 
        operator()(const blitz::Array<uint16_t,2>& src, 
            blitz::Array<uint16_t,2>& dst) const 
          { operator()<uint16_t>(src, dst); }
        void 
        operator()(const blitz::Array<double,2>& src, 
            blitz::Array<uint16_t,2>& dst) const 
          { operator()<double>(src, dst); }

        /**
          * @brief Extract the LBP code of a 2D blitz::Array at the given 
          *   location, and return it.
          */
        template <typename T> 
        uint16_t operator()(const blitz::Array<T,2>& src, int y, int x) const;
        uint16_t 
        operator()(const blitz::Array<uint8_t,2>& src, int y, int x) const 
          { return operator()<uint8_t>( src, y, x); }
        uint16_t 
        operator()(const blitz::Array<uint16_t,2>& src, int y, int x) const 
          { return operator()<uint16_t>( src, y, x); }
        uint16_t 
        operator()(const blitz::Array<double,2>& src, int y, int x) const 
          { return operator()<double>( src, y, x); }

        /**
          * @brief Get the required shape of the dst output blitz array, 
          *   before calling the operator() method.
          */
        template <typename T>
        const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<T,2>& src) const;
        const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<uint8_t,2>& src) const
          { return getLBPShape<uint8_t>(src); }
        const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<uint16_t,2>& src) const
          { return getLBPShape<uint16_t>(src); }
        const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<double,2>& src) const
          { return getLBPShape<double>(src); }

    	private:
        /**
          * @brief Extract the LBP code of a 2D blitz::Array at the given 
          *   location, and return it, without performing any check.
          */
        template <typename T, bool circular>
        uint16_t processNoCheck(const blitz::Array<T,2>& src, int y, int x) 
          const;

		    /**
    		  * @brief Initialize the conversion table for rotation invariant and
          * uniform LBP patterns
          */
		    virtual void init_lut_RI();
    	  virtual void init_lut_U2();
        virtual void init_lut_U2RI();
        virtual void init_lut_add_average_bit();
        virtual void init_lut_normal();
    };

    template <typename T>
    void bob::ip::LBP8R::operator()(const blitz::Array<T,2>& src,  
      blitz::Array<uint16_t,2>& dst) const
    {
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      bob::core::array::assertSameShape(dst, getLBPShape(src) );
      if( m_circular)
      {
        for(int y=0; y<dst.extent(0); ++y)
          for(int x=0; x<dst.extent(1); ++x)
            dst(y,x) = 
              bob::ip::LBP8R::processNoCheck<T,true>(src,
                static_cast<int>(ceil(m_R))+y, static_cast<int>(ceil(m_R))+x);
      }
      else
      {
        for(int y=0; y<dst.extent(0); ++y)
          for(int x=0; x<dst.extent(1); ++x)
            dst(y,x) = 
              bob::ip::LBP8R::processNoCheck<T,false>(src,
                static_cast<int>(m_R_rect)+y, static_cast<int>(m_R_rect)+x);
      }
    }
    
    template <typename T> 
    uint16_t bob::ip::LBP8R::operator()(const blitz::Array<T,2>& src, 
      int yc, int xc) const
    {
      bob::core::array::assertZeroBase(src);
      if( m_circular)
      {
        if( yc<ceil(m_R) )
          throw ParamOutOfBoundaryError("yc", false, yc, ceil(m_R));
        if( yc>=src.extent(0)-ceil(m_R) )
          throw ParamOutOfBoundaryError("yc", true, yc, src.extent(0)-ceil(m_R)-1);
        if( xc<ceil(m_R) )
          throw ParamOutOfBoundaryError("xc", false, xc, ceil(m_R));
        if( xc>=src.extent(1)-ceil(m_R) )
          throw ParamOutOfBoundaryError("xc", true, xc, src.extent(1)-ceil(m_R)-1);
        return bob::ip::LBP8R::processNoCheck<T,true>( src, yc, xc);
      }
      else
      {
        if( yc<m_R_rect )
          throw ParamOutOfBoundaryError("yc", false, yc, m_R_rect);
        if( yc>=src.extent(0)-m_R_rect )
          throw ParamOutOfBoundaryError("yc", true, yc, src.extent(0)-m_R_rect-1);
        if( xc<m_R_rect )
          throw ParamOutOfBoundaryError("xc", false, xc, m_R_rect);
        if( xc>=src.extent(1)-m_R_rect )
          throw ParamOutOfBoundaryError("xc", true, xc, src.extent(1)-m_R_rect-1);
        return bob::ip::LBP8R::processNoCheck<T,false>( src, yc, xc);
      }
    }
    
    template <typename T, bool circular> 
    uint16_t bob::ip::LBP8R::processNoCheck( const blitz::Array<T,2>& src,
      int yc, int xc) const
    {
      double tab[8];
      if(circular)
      {
        const double R_sqrt2 = m_R / sqrt(2);
        tab[0] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc-R_sqrt2,xc-R_sqrt2);
        tab[1] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc-m_R,xc);
        tab[2] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc-R_sqrt2,xc+R_sqrt2);
        tab[3] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc,xc+m_R);
        tab[4] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc+R_sqrt2,xc+R_sqrt2);
        tab[5] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc+m_R,xc);
        tab[6] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc+R_sqrt2,xc-R_sqrt2);
        tab[7] = bob::sp::detail::bilinearInterpolationNoCheck(src,yc,xc-m_R);
      }
      else
      {
        tab[0] = static_cast<double>(src(yc-m_R_rect,xc-m_R_rect));
        tab[1] = static_cast<double>(src(yc-m_R_rect,xc));
        tab[2] = static_cast<double>(src(yc-m_R_rect,xc+m_R_rect));
        tab[3] = static_cast<double>(src(yc,xc+m_R_rect));
        tab[4] = static_cast<double>(src(yc+m_R_rect,xc+m_R_rect));
        tab[5] = static_cast<double>(src(yc+m_R_rect,xc));
        tab[6] = static_cast<double>(src(yc+m_R_rect,xc-m_R_rect));
        tab[7] = static_cast<double>(src(yc,xc-m_R_rect));
      }

      const T center = src(yc,xc);
      const double cmp_point = (m_to_average ? 
        ( 0.1111111111 * (tab[0] + tab[1] + tab[2] + tab[3] + tab[4] + tab[5] + tab[6] + tab[7] + center)) : center);

      uint16_t lbp = 0;
      // lbp = lbp << 1; // useless
      if(m_eLBP_type == 0) // regular LBP
      {
        if(tab[0] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[1] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[2] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[3] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[4] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[5] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[6] >= cmp_point) ++lbp;
        lbp = lbp << 1;
        if(tab[7] >= cmp_point) ++lbp;
      } 

      if (m_eLBP_type == 1) // transitional LBP
      {
        for(int i=0; i<=7; i++)
        {
          lbp = lbp << 1;
          if(i==7)
          {
            if(tab[i] >= tab[0]) ++lbp;
          }
          else    
            if(tab[i] >= tab[i+1]) ++lbp;
        }
      }

      if (m_eLBP_type == 2) //directional coded LBP
      {
        for(int i=0; i<=3; i++)
        {
          lbp = lbp << 2;
          if((tab[i] >= cmp_point && tab[i+4] >= cmp_point) || (tab[i] < cmp_point && tab[i+4] < cmp_point))
            if (fabs(tab[i]-cmp_point) > fabs(tab[i+4]-cmp_point)) lbp+=3;
            else lbp+=2;
          else
            if (fabs(tab[i]-cmp_point) > fabs(tab[i+4]-cmp_point)) lbp+=0;
            else lbp+=1;
        }
      }

      if(m_add_average_bit && !m_rotation_invariant && !m_uniform)
      {
        lbp = lbp << 1;
        if(center > cmp_point) ++lbp;
      }

      return m_lut_current(lbp);
    }

    template <typename T>
    const blitz::TinyVector<int,2> 
    bob::ip::LBP8R::getLBPShape(const blitz::Array<T,2>& src) const
    {
      blitz::TinyVector<int,2> res;
      res(0) = std::max(0, src.extent(0)-2*static_cast<int>(ceil(m_R)));
      res(1) = std::max(0, src.extent(1)-2*static_cast<int>(ceil(m_R)));
      return res;
    }

  }
}

#endif /* BOB5SPRO_IP_LBP8R_H */
