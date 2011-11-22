/**
 * @file cxx/sp/sp/FFT2DNaive.h
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 2D Discrete Fourier Transform
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

#ifndef TORCH5SPRO_SP_FFT2D_NAIVE_H
#define TORCH5SPRO_SP_FFT2D_NAIVE_H

#include <complex>
#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp { namespace detail {

    /**
      * @brief This class implements a naive 1D Discrete Fourier Transform.
      */
    class FFT2DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        FFT2DNaiveAbstract( const int height, const int width);

        /**
          * @brief Destructor
          */
        virtual ~FFT2DNaiveAbstract();

        /**
          * @brief process an array by applying the FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
          blitz::Array<std::complex<double>,2>& dst) = 0;

        /**
          * @brief Reset the FFT2DNaive object for the given 2D shape
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
        blitz::Array<std::complex<double>,1> m_wsave_h; 
        blitz::Array<std::complex<double>,1> m_wsave_w;
    };


    /**
      * @brief This class implements a naive direct 1D Discrete Fourier 
      * Transform
      */
    class FFT2DNaive: public FFT2DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        FFT2DNaive( const int height, const int width);

        /**
          * @brief process an array by applying the direct FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
          blitz::Array<std::complex<double>,2>& dst);
      
      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
          blitz::Array<std::complex<double>,2>& dst);
    };


    /**
      * @brief This class implements a naive inverse 1D Discrete Fourier 
      * Transform 
      */
    class IFFT2DNaive: public FFT2DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IFFT2DNaive( const int height, const int width);

        /**
          * @brief process an array by applying the inverse FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,2>& src, 
          blitz::Array<std::complex<double>,2>& dst);

      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<std::complex<double>,2>& src,
          blitz::Array<std::complex<double>,2>& dst);
    };

  }}
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_FFT2D_NAIVE_H */
