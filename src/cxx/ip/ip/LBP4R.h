/**
 * @file cxx/ip/ip/LBP4R.h
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines a function to compute the LBP4R.
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef TORCH5SPRO_IP_LBP4R_H
#define TORCH5SPRO_IP_LBP4R_H

#include <blitz/array.h>
#include <algorithm>
#include "core/array_assert.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/LBP.h"
#include "sp/interpolate.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
      * @brief This class allows to extract Local Binary Pattern-like features
      *   based on 4 neighbour pixels.
      *   For more information, please refer to the following article:
      *     "Face Recognition with Local Binary Patterns", from T. Ahonen,
      *     A. Hadid and M. Pietikainen
      *     in the proceedings of the European Conference on Computer Vision
      *     (ECCV'2004), p. 469-481
      */
    class LBP4R: public LBP
    {
      public:
        /**
          * @brief Constructor
          */
        LBP4R(const double R=1., const bool circular=false, 
            const bool to_average=false, const bool add_average_bit=false,
            const bool uniform=false, const bool rotation_invariant=false);

        /**
          * @brief Destructor
          */
        virtual ~LBP4R() { }

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
    		virtual void init_lut_normal();
    		virtual void init_lut_add_average_bit();
    };

    template <typename T>
    void Torch::ip::LBP4R::operator()(const blitz::Array<T,2>& src,  
      blitz::Array<uint16_t,2>& dst) const
    {
      Torch::core::array::assertZeroBase(src);
      Torch::core::array::assertZeroBase(dst);
      Torch::core::array::assertSameShape(dst, getLBPShape(src) );
      if( m_circular)
      {
        for(int y=0; y<dst.extent(0); ++y)
          for(int x=0; x<dst.extent(1); ++x)
            dst(y,x) = 
              Torch::ip::LBP4R::processNoCheck<T,true>(src, 
                static_cast<int>(ceil(m_R))+y, static_cast<int>(ceil(m_R))+x);
      }
      else
      {
        for(int y=0; y<dst.extent(0); ++y)
          for(int x=0; x<dst.extent(1); ++x)
            dst(y,x) = 
              Torch::ip::LBP4R::processNoCheck<T,false>(src, 
                static_cast<int>(m_R_rect)+y, static_cast<int>(m_R_rect)+x);
      }
    }
    
    template <typename T> 
    uint16_t Torch::ip::LBP4R::operator()(const blitz::Array<T,2>& src, 
      int yc, int xc) const
    {
      Torch::core::array::assertZeroBase(src);
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
        return Torch::ip::LBP4R::processNoCheck<T,true>( src, yc, xc);
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
        return Torch::ip::LBP4R::processNoCheck<T,false>( src, yc, xc);
      }
    }
    
    template <typename T, bool circular> 
    uint16_t Torch::ip::LBP4R::processNoCheck( const blitz::Array<T,2>& src,
      int yc, int xc) const
    {
      double tab[4];
      if(circular)
      {
        tab[0] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc-m_R,xc);
        tab[1] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc,xc+m_R);
        tab[2] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc+m_R,xc);
        tab[3] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc,xc-m_R);
      }
      else
      {
        tab[0] = static_cast<double>(src(yc-m_R_rect,xc));
        tab[1] = static_cast<double>(src(yc,xc+m_R_rect));
        tab[2] = static_cast<double>(src(yc+m_R_rect,xc));
        tab[3] = static_cast<double>(src(yc,xc-m_R_rect));
      }
  
      const T center = src(yc,xc);
      const double cmp_point = (m_to_average ? 
        ( 0.2 * (tab[0] + tab[1] + tab[2] + tab[3] + center + 0.0)) : center);

      uint16_t lbp = 0;
      // lbp = lbp << 1; // useless
      if(tab[0] >= cmp_point) ++lbp;
      lbp = lbp << 1;
      if(tab[1] >= cmp_point) ++lbp;
      lbp = lbp << 1;
      if(tab[2] >= cmp_point) ++lbp;
      lbp = lbp << 1;
      if(tab[3] >= cmp_point) ++lbp;
      if(m_add_average_bit && !m_rotation_invariant && !m_uniform)
      {
        lbp = lbp << 1;
        if(center > cmp_point) ++lbp;
      }

      return m_lut_current(lbp);
    }

    template <typename T>
    const blitz::TinyVector<int,2> 
    Torch::ip::LBP4R::getLBPShape(const blitz::Array<T,2>& src) const
    {
      blitz::TinyVector<int,2> res;
      res(0) = std::max(0, src.extent(0)-2*static_cast<int>(ceil(m_R)));
      res(1) = std::max(0, src.extent(1)-2*static_cast<int>(ceil(m_R)));
      return res;
    }

  }
}

#endif /* TORCH5SPRO_IP_LBP4R_H */
