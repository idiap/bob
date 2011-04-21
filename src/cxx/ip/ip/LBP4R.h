/**
 * @file src/cxx/ip/ip/LBP4R.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines a function to compute the LBP4R.
 */

#ifndef TORCH5SPRO_IP_LBP4R_H
#define TORCH5SPRO_IP_LBP4R_H

#include <blitz/array.h>
#include "core/array_assert.h"
#include "core/cast.h"
#include "ip/Exception.h"
#include "ip/LBP.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    class LBP4R: public LBP
    {
      public:
        LBP4R(const int R=1, const bool to_average=false, 
            const bool add_average_bit=false, const bool uniform=false, 
            const bool rotation_invariant=false); 

        virtual ~LBP4R() { }


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
        template <typename T>
        uint16_t processNoCheck(const blitz::Array<T,2>& src, int y, int x) 
          const;

		    /////////////////////////////////////////////////////////////////
    		// Initialize the conversion table for rotation invariant and uniform LBP patterns
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
      for(int y=0; y<dst.extent(0); ++y)
        for(int x=0; x<dst.extent(1); ++x)
          dst(y,x) = Torch::ip::LBP4R::processNoCheck(src, m_R+y, m_R+x);
    }
    
    template <typename T> 
    uint16_t Torch::ip::LBP4R::operator()(const blitz::Array<T,2>& src, 
      int yc, int xc) const
    {
      // TODO: check inputs (xc, yc, etc.)
      if( yc<m_R || yc>=src.extent(0)-m_R || xc<m_R || xc>=src.extent(1)-m_R)
        throw Torch::ip::Exception();
      
      return Torch::ip::LBP4R::processNoCheck( src, yc, xc);
    }
    
    template <typename T> 
    uint16_t Torch::ip::LBP4R::processNoCheck( const blitz::Array<T,2>& src,
      int yc, int xc) const
    {
      T tab[4];
      tab[0] = src(yc-m_R,xc);
      tab[1] = src(yc,xc+m_R);
      tab[2] = src(yc+m_R,xc);
      tab[3] = src(yc,xc-m_R);
  
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
      res(0) = std::max(0, src.extent(0)-2*m_R);
      res(1) = std::max(0, src.extent(1)-2*m_R);
      return res;
    }

  }
}

#endif /* TORCH5SPRO_IP_LBP4R_H */
