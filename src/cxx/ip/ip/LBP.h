/**
 * @file cxx/ip/ip/LBP.h
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines classes to compute LBP and variants
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

#ifndef TORCH5SPRO_IP_LBP_H
#define TORCH5SPRO_IP_LBP_H

#include <blitz/array.h>
#include <stdint.h> // uint16_t declaration

namespace Torch {
/**
 * \ingroup libip_api
 * @{
 *
 */
  namespace ip {

    /**
      * @brief This class is an abstraction for all the Local Binary Patterns
      *   variants. For more information, please refer to the following 
      *   article:
      *     "Face Recognition with Local Binary Patterns", from T. Ahonen,
      *     A. Hadid and M. Pietikainen
      *     in the proceedings of the European Conference on Computer Vision
      *     (ECCV'2004), p. 469-481
      */
    class LBP
    {
      public:
        /**
          * @brief Constructor
          */
        LBP(const int P, const double R=1., const bool circular=false,
          const bool to_average=false, const bool add_average_bit=false, 
          const bool uniform=false, const bool rotation_invariant=false);

        /**
          * @brief Destructor
          */
        virtual ~LBP() { }

		    /**
          * @brief Return the maximum number of labels for the current LBP 
          *   variant
          */
  		  virtual int getMaxLabel() const = 0;

        /**
          * @brief Accessors
          */
        double getRadius() const { return m_R; }
        int getNNeighbours() const { return m_P; }
        bool getCircular() const { return m_circular; }
        bool getToAverage() const { return m_to_average; }
        bool getAddAverageBit() const { return m_add_average_bit; }
        bool getUniform() const { return m_uniform; }
        bool getRotationInvariant() const { return m_rotation_invariant; }

        /**
          * @brief Mutators
          */
        void setRadius(const double R) 
          { m_R = R; updateR(); }
        void setCircular(const bool circ) 
          { m_circular = circ; init_lut_current(); }
        void setToAverage(const bool to_a) 
          { m_to_average = to_a; init_lut_current(); }
        void setAddAverageBit(const bool add_a_b) 
          { m_add_average_bit = add_a_b; init_lut_current(); }
        void setUniform(const bool unif) 
          { m_uniform = unif; init_lut_current(); }
        void setRotationInvariant(const bool rot_i) 
          { m_rotation_invariant = rot_i; init_lut_current(); }

        /**
          * @brief Extract LBP features from a 2D blitz::Array, and save 
          *   the resulting LBP codes in the dst 2D blitz::Array.
          */
        virtual void 
        operator()(const blitz::Array<uint8_t,2>& src, 
          blitz::Array<uint16_t,2>& dst) const = 0;
        virtual void 
        operator()(const blitz::Array<uint16_t,2>& src, 
          blitz::Array<uint16_t,2>& dst) const = 0;
        virtual void 
        operator()(const blitz::Array<double,2>& src, 
          blitz::Array<uint16_t,2>& dst) const = 0;

        /**
          * @brief Extract the LBP code of a 2D blitz::Array at the given 
          *   location, and return it.
          */
        virtual uint16_t 
        operator()(const blitz::Array<uint8_t,2>& src, 
          int y, int x) const = 0;
        virtual uint16_t 
        operator()(const blitz::Array<uint16_t,2>& src, 
          int y, int x) const = 0;
        virtual uint16_t 
        operator()(const blitz::Array<double,2>& src, 
          int y, int x) const = 0;

        /**
          * @brief Get the required shape of the dst output blitz array, 
          *   before calling the operator() method.
          */
        virtual const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<uint8_t,2>& src) const = 0;
        virtual const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<uint16_t,2>& src) const = 0;
        virtual const blitz::TinyVector<int,2> 
        getLBPShape(const blitz::Array<double,2>& src) const = 0;

    	protected:
		    /**
          * @brief Initialize the conversion table for rotation invariant and 
          *   uniform LBP patterns
          */
		    virtual void init_lut_RI() = 0;
    		virtual void init_lut_U2() = 0;
    		virtual void init_lut_U2RI()= 0;
    		virtual void init_lut_add_average_bit()= 0;
    		virtual void init_lut_normal()= 0;
    		void init_lut_current();
		    /**
          * @brief Initialize all the conversion tables 
          */
        void init_luts();

		    /**
          * @brief Compute the current integer value of the radius in case 
          *   of a non-circular LBP variant
          */ 
        inline void updateR() { m_R_rect = static_cast<int>(floor(m_R+0.5)); }

        /**
          * @brief Attributes
          */
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
