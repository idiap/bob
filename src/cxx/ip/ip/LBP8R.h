/**
 * @file src/cxx/ip/ip/LBP8R.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to compute the LBP8R.
 */

#ifndef TORCH5SPRO_IP_LBP8R_H
#define TORCH5SPRO_IP_LBP8R_H

#include <blitz/array.h>
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

    class LBP8R: public LBP
    {
      public:
        LBP8R(const double R=1., const bool circular=false, 
            const bool to_average=false, const bool add_average_bit=false, 
            const bool uniform=false, const bool rotation_invariant=false);

        virtual ~LBP8R() { }


		    // Get the maximum possible label
  		  virtual int getMaxLabel() const;

        template <typename T> 
        void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<uint16_t,2>& dst) const;

        template <typename T> 
        uint16_t operator()(const blitz::Array<T,2>& src, int y, int x) const;

        template <typename T>
        const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<T,2>& src) const;

    	private:
        template <typename T, bool circular>
        uint16_t processNoCheck(const blitz::Array<T,2>& src, int y, int x) 
          const;

		    /////////////////////////////////////////////////////////////////
    		// Initialize the conversion table for rotation invariant and uniform LBP patterns
		    virtual void init_lut_RI();
    	  virtual void init_lut_U2();
        virtual void init_lut_U2RI();
        virtual void init_lut_add_average_bit();
        virtual void init_lut_normal();
    };

    template <typename T>
    void Torch::ip::LBP8R::operator()(const blitz::Array<T,2>& src,  
      blitz::Array<uint16_t,2>& dst) const
    {
      Torch::core::array::assertSameShape(dst, getLBPShape(src) );
      if( m_circular)
      {
        for(int y=0; y<dst.extent(0); ++y)
          for(int x=0; x<dst.extent(1); ++x)
            dst(y,x) = 
              Torch::ip::LBP8R::processNoCheck<T,true>(src, m_R+y, m_R+x);
      }
      else
      {
        for(int y=0; y<dst.extent(0); ++y)
          for(int x=0; x<dst.extent(1); ++x)
            dst(y,x) = 
              Torch::ip::LBP8R::processNoCheck<T,false>(src, m_R+y, m_R+x);
      }
    }
    
    template <typename T> 
    uint16_t Torch::ip::LBP8R::operator()(const blitz::Array<T,2>& src, 
      int yc, int xc) const
    {
      // TODO: check inputs (xc, yc, etc.)
      if( m_circular)
      {
        if( yc<m_R_rect || yc>=src.extent(0)-m_R_rect || 
            xc<m_R_rect || xc>=src.extent(1)-m_R_rect)
          throw Torch::ip::Exception();
        return Torch::ip::LBP8R::processNoCheck<T,true>( src, yc, xc);
      }
      else
      {
        if( yc<m_R || yc>=src.extent(0)-m_R || 
            xc<m_R || xc>=src.extent(1)-m_R)
          throw Torch::ip::Exception();
        return Torch::ip::LBP8R::processNoCheck<T,false>( src, yc, xc);
      }
    }
    
    template <typename T, bool circular> 
    uint16_t Torch::ip::LBP8R::processNoCheck( const blitz::Array<T,2>& src,
      int yc, int xc) const
    {
      double tab[8];
      if(circular)
      {
        const double R_sqrt2 = m_R / sqrt(2);
        tab[0] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc-R_sqrt2,xc-R_sqrt2);
        tab[1] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc-m_R,xc);
        tab[2] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc-R_sqrt2,xc+R_sqrt2);
        tab[3] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc,xc+m_R);
        tab[4] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc+R_sqrt2,xc+R_sqrt2);
        tab[5] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc+m_R,xc);
        tab[6] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc+R_sqrt2,xc-R_sqrt2);
        tab[7] = Torch::sp::detail::bilinearInterpolationNoCheck(src,yc,xc-m_R);
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
      if(m_add_average_bit && !m_rotation_invariant && !m_uniform)
      {
        lbp = lbp << 1;
        if(center > cmp_point) ++lbp;
      }

      return m_lut_current(lbp);
    }

    template <typename T>
    const blitz::TinyVector<int,2> 
    Torch::ip::LBP8R::getLBPShape(const blitz::Array<T,2>& src) const
    {
      blitz::TinyVector<int,2> res;
      res(0) = std::max(0, src.extent(0)-2*m_R);
      res(1) = std::max(0, src.extent(1)-2*m_R);
      return res;
    }

  }
}

#endif /* TORCH5SPRO_IP_LBP8R_H */
