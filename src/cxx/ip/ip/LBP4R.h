/**
 * @file src/cxx/ip/ip/LBP4R.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to compute the LBP4R.
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

        /**
          * @brief Extract the LBP code of a 2D blitz::Array at the given 
          *   location, and return it.
          */
        template <typename T> 
        uint16_t operator()(const blitz::Array<T,2>& src, int y, int x) const;

        /**
          * @brief Get the required shape of the dst output blitz array, 
          *   before calling the operator() method.
          */
        template <typename T>
        const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<T,2>& src) const;

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
                static_cast<int>(ceil(m_R))+y, static_cast<int>(ceil(m_R))+x);
      }
    }
    
    template <typename T> 
    uint16_t Torch::ip::LBP4R::operator()(const blitz::Array<T,2>& src, 
      int yc, int xc) const
    {
      // TODO: check inputs (xc, yc, etc.)
      if( m_circular)
      {
        if( yc<ceil(m_R) || yc>=src.extent(0)-ceil(m_R) || 
            xc<ceil(m_R) || xc>=src.extent(1)-ceil(m_R) )
          throw Torch::ip::Exception();
        return Torch::ip::LBP4R::processNoCheck<T,true>( src, yc, xc);
      }
      else
      {
        if( yc<m_R || yc>=src.extent(0)-m_R || 
            xc<m_R || xc>=src.extent(1)-m_R)
          throw Torch::ip::Exception();
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
