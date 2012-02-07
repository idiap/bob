/**
 * @file cxx/sp/sp/FFT2D.h
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 2D Fast Fourier Transform using FFTPACK
 * functions
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_SP_FFT2D_H
#define BOB_SP_FFT2D_H

#include <complex>
#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp {

    /**
      * @brief This class implements a Discrete Fourier Transform based on the
      * Netlib FFTPACK library. It is used as a base class for FFT2D and 
      * IFFT2D classes.
      */
    class FFT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        FFT2DAbstract( const int height, const int width);

        /**
          * @brief Destructor
          */
        virtual ~FFT2DAbstract();

        /**
          * @brief process an array by applying the FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
          blitz::Array<std::complex<double>,2>& dst) = 0;

        /**
          * @brief Reset the FFT2D object for the given 2D shape
          */
        void reset(const int height, const int width);

        /**
          * @brief Get the current height of the FFT2D object
          */
        inline const int getHeight() { return m_height; }

        /**
          * @brief Get the current width of the FFT2D object
          */
        inline const int getWidth() { return m_width; }

      private:
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
    };


    /**
      * @brief This class implements a direct 2D Discrete Fourier Transform 
      * based on the FFTPACK library
      */
    class FFT2D: public FFT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        FFT2D( const int height, const int width);

        /**
          * @brief process an array by applying the direct FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
          blitz::Array<std::complex<double>,2>& dst);
    };


    /**
      * @brief This class implements a inverse 2D Discrete Fourier Transform 
      * based on the FFTPACK library
      */
    class IFFT2D: public FFT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        IFFT2D( const int height, const int width);

        /**
          * @brief process an array by applying the inverse FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
          blitz::Array<std::complex<double>,2>& dst);
    };

  }
/**
 * @}
 */
}

#endif /* BOB_SP_FFT2D_H */
