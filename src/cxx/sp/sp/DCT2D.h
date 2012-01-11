/**
 * @file cxx/sp/sp/DCT2D.h
 * @date Tue Apr 5 19:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK
 * functions
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

#ifndef BOB5SPRO_SP_DCT2D_H
#define BOB5SPRO_SP_DCT2D_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp {

    /**
      * @brief This class implements a Discrete Cosine Transform based on the
      * Netlib FFTPACK library. It is used as a base class for DCT2D and 
      * IDCT2D classes.
      */
    class DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        DCT2DAbstract( const int height, const int width);

        /**
          * @brief Destructor
          */
        virtual ~DCT2DAbstract();

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst) = 0;

        /**
          * @brief Reset the DCT2D object for the given 2D shape
          */
        void reset(const int height, const int width);

        /**
          * @brief Get the current height of the DCT2D object
          */
        inline const int getHeight() { return m_height; }

        /**
          * @brief Get the current width of the DCT2D object
          */
        inline const int getWidth() { return m_width; }

      private:
        /**
          * @brief Initialize the normalization factors
          */
        void initNormFactors();

        /**
          * @brief Call the initialization procedures
          */
        void reset();

        /**
          * @brief Deallocate memory
          */
        void cleanup();

      protected:
        /**
          * Private attributes
          */
        int m_height;
        int m_width;

        /**
          * Normalization factors
          */
        double m_sqrt_1h;
        double m_sqrt_2h;
        double m_sqrt_1w;
        double m_sqrt_2w;
    };


    /**
      * @brief This class implements a direct 2D Discrete Cosine Transform 
      * based on the FFTPACK library
      */
    class DCT2D: public DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT2D( const int height, const int width);

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);
    };


    /**
      * @brief This class implements a inverse 2D Discrete Cosine Transform 
      * based on the FFTPACK library
      */
    class IDCT2D: public DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        IDCT2D( const int height, const int width);

        /**
          * @brief process an array by applying the inverse DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);
    };

  }
/**
 * @}
 */
}

#endif /* BOB5SPRO_SP_DCT2D_H */
