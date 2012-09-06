/**
 * @file bob/sp/DCT2D.h
 * @date Tue Apr 5 19:18:23 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTW
 * functions
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_SP_DCT2D_H
#define BOB_SP_DCT2D_H

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
      * FFTW library. It is used as a base class for DCT2D and IDCT2D classes.
      */
    class DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        DCT2DAbstract( const size_t height, const size_t width);

        /**
          * @brief Copy constructor
          */
        DCT2DAbstract( const DCT2DAbstract& other);

        /**
          * @brief Destructor
          */
        virtual ~DCT2DAbstract();

        /**
          * @brief Assignment operator
          */
        const DCT2DAbstract& operator=(const DCT2DAbstract& other);

        /**
          * @brief Equal operator
          */
        bool operator==(const DCT2DAbstract& other) const;

        /**
          * @brief Not equal operator
          */
        bool operator!=(const DCT2DAbstract& other) const;

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst) = 0;

        /**
          * @brief Reset the DCT2D object for the given 2D shape
          */
        void reset(const size_t height, const size_t width);

        /**
          * @brief Getters
          */
        size_t getHeight() const { return m_height; }
        size_t getWidth() const { return m_width; }

        /**
          * @brief Setters
          */
        void setHeight(const size_t height);
        void setWidth(const size_t width);

      private:
        /**
          * @brief Initialize the normalization factors
          */
        void initNormFactors();

        /**
          * @brief Call the initialization procedures
          */
        void reset();

      protected:
        /**
          * Private attributes
          */
        size_t m_height;
        size_t m_width;

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
      * based on the FFTW library
      */
    class DCT2D: public DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT2D( const size_t height, const size_t width);

        /**
          * @brief Copy constructor
          */
        DCT2D( const DCT2D& other);

        /**
          * @brief Destructor
          */
        virtual ~DCT2D();

        /**
          * @brief Assignment operator
          */
        const DCT2D& operator=(const DCT2D& other);

        /**
          * @brief Equal operator
          */
        bool operator==(const DCT2D& other) const;

        /**
          * @brief Not equal operator
          */
        bool operator!=(const DCT2D& other) const;

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);
    };


    /**
      * @brief This class implements a inverse 2D Discrete Cosine Transform 
      * based on the FFTW library
      */
    class IDCT2D: public DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        IDCT2D( const size_t height, const size_t width);

        /**
          * @brief Copy constructor
          */
        IDCT2D( const IDCT2D& other);

        /**
          * @brief Destructor
          */
        virtual ~IDCT2D();

        /**
          * @brief Assignment operator
          */
        const IDCT2D& operator=(const IDCT2D& other);

        /**
          * @brief Equal operator
          */
        bool operator==(const IDCT2D& other) const;

        /**
          * @brief Not equal operator
          */
        bool operator!=(const IDCT2D& other) const;

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

#endif /* BOB_SP_DCT2D_H */
