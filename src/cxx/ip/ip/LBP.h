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
        // TODO Make the radius R a float?
        LBP(const int P, const int R=1, const bool to_average=false, 
          const bool add_average_bit=false, const bool uniform=false, 
          const bool rotation_invariant=false);

        virtual ~LBP() { }

		    // Get the maximum possible label
  		  virtual int getMaxLabel() const = 0;

        void setRadius(const int R) { m_R = R; }
        int getRadius() const { return m_R; }
        int getNNeighbours() const { return m_P; }

/*
        template<typename T> virtual void 
        operator()(const blitz::Array<T,2>& src, 
          blitz::Array<uint16_t,2>& dst) const = 0;
*/
    	protected:

		    /////////////////////////////////////////////////////////////////
    		// Initialize the conversion table for rotation invariant and uniform LBP patterns
		    //void			init_lut_RI();
    		//void			init_lut_U2();
    		//void			init_lut_U2RI();

        int m_P;
        int m_R;
        bool m_to_average;
        bool m_add_average_bit;
        bool m_uniform;
        bool m_rotation_invariant;

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
