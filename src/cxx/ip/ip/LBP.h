/**
 * @file src/cxx/ip/ip/LBP.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This file defines classes to compute LBP and variants
 */

#ifndef TORCH5SPRO_IP_LBP_H
#define TORCH5SPRO_IP_LBP_H

#include <blitz/array.h>
#include "core/array_assert.h"
#include "core/cast.h"
#include "ip/Exception.h"

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    class LBP
    {
      public:
        LBP(const int P, const double R=1., const bool circular=true,
          const bool to_average=false, const bool add_average_bit=false, 
          const bool uniform=false, const bool rotation_invariant=false);

        virtual ~LBP() { }

		    // Get the maximum possible label
  		  virtual int getMaxLabel() const = 0;

        void setRadius(const double R) { m_R = R; updateR(); }
        double getRadius() const { return m_R; }
        int getNNeighbours() const { return m_P; }

    	protected:
		    /////////////////////////////////////////////////////////////////
    		// Initialize the conversion table for rotation invariant and uniform LBP patterns
		    virtual void init_lut_RI() = 0;
    		virtual void init_lut_U2() = 0;
    		virtual void init_lut_U2RI()= 0;
    		virtual void init_lut_add_average_bit()= 0;
    		virtual void init_lut_normal()= 0;
    		void init_lut_current();

        inline void updateR() { m_R_rect = static_cast<int>(floor(m_R+0.5)); }

        int m_P;
        double m_R;
        bool m_circular;
        bool m_to_average;
        bool m_add_average_bit;
        bool m_uniform;
        bool m_rotation_invariant;
        int m_R_rect;

        blitz::Array<uint16_t,1> m_lut_RI;
        blitz::Array<uint16_t,1> m_lut_U2;
        blitz::Array<uint16_t,1> m_lut_U2RI;
        blitz::Array<uint16_t,1> m_lut_add_average_bit;
        blitz::Array<uint16_t,1> m_lut_normal;

        blitz::Array<uint16_t,1> m_lut_current;
    };

  }
}

#endif /* TORCH5SPRO_IP_LBP_H */
