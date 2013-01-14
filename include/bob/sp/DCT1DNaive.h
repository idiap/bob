/**
 * @file bob/sp/DCT1DNaive.h
 * @date Thu Apr 7 17:02:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Discrete Cosine Transform
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_SP_DCT1D_NAIVE_H
#define BOB_SP_DCT1D_NAIVE_H

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
    class DCT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */
        DCT1DNaiveAbstract( const int length);

        /**
          * @brief Destructor
          */
        virtual ~DCT1DNaiveAbstract();

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst) = 0;

        /**
          * @brief Reset the DCT1DNaive object for the given 1D shape
          */
        void reset(const int length);

        /**
          * @brief Get the current height of the DCT1D object
          */
        inline const int getLength() { return m_length; }

      private:
        /**
          * @brief Initialize the normalization factors
          */
        void initNormFactors();

        /**
          * @brief Initialize the working array
          */
        void initWorkingArray();

        /**
          * @brief Call the initialization procedures
          */
        void reset();

      protected:
        /**
          * Private attributes
          */
        int m_length;

        /**
          * Working array
          */
        blitz::Array<double,1> m_wsave; 

        /**
          * Normalization factors
          */
        double m_sqrt_1l;
        double m_sqrt_2l;
    };


    /**
      * @brief This class implements a naive direct 1D Discrete Cosine 
      * Transform
      */
    class DCT1DNaive: public DCT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT1DNaive( const int length);

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst);
      
      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<double,1>& src,
          blitz::Array<double,1>& dst);
    };


    /**
      * @brief This class implements a naive inverse 1D Discrete Cosine 
      * Transform 
      */
    class IDCT1DNaive: public DCT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IDCT1DNaive( const int length);

        /**
          * @brief process an array by applying the inverse DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst);

      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<double,1>& src,
          blitz::Array<double,1>& dst);
    };

  }}
/**
 * @}
 */
}

#endif /* BOB_SP_DCT1D_NAIVE_H */
