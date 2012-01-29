/**
 * @file cxx/sp/sp/DCT2DNaive.h
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Discrete Cosine Transform
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

#ifndef BOB_SP_DCT2D_NAIVE_H
#define BOB_SP_DCT2D_NAIVE_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp { namespace detail {

    /**
      * @brief This class implements a naive 1D Discrete Cosine Transform.
      */
    class DCT2DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        DCT2DNaiveAbstract( const int height, const int width);

        /**
          * @brief Destructor
          */
        virtual ~DCT2DNaiveAbstract();

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst) = 0;

        /**
          * @brief Reset the DCT2DNaive object for the given 2D shape
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
          * @brief Initialize the working arrays
          */
        void initWorkingArrays();

        /**
          * @brief Call the initialization procedures
          */
        void reset();

      protected:
        /**
          * Private attributes
          */
        int m_height;
        int m_width;

        /**
          * Working array
          */
        blitz::Array<double,1> m_wsave_h; 
        blitz::Array<double,1> m_wsave_w;

        /**
          * Normalization factors
          */
        double m_sqrt_1h;
        double m_sqrt_2h;
        double m_sqrt_1w;
        double m_sqrt_2w;
    };


    /**
      * @brief This class implements a naive direct 1D Discrete Cosine 
      * Transform
      */
    class DCT2DNaive: public DCT2DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT2DNaive( const int height, const int width);

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);
      
      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<double,2>& src,
          blitz::Array<double,2>& dst);
    };


    /**
      * @brief This class implements a naive inverse 1D Discrete Cosine 
      * Transform 
      */
    class IDCT2DNaive: public DCT2DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IDCT2DNaive( const int height, const int width);

        /**
          * @brief process an array by applying the inverse DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);

      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<double,2>& src,
          blitz::Array<double,2>& dst);
    };

  }}
/**
 * @}
 */
}

#endif /* BOB_SP_DCT2D_NAIVE_H */
