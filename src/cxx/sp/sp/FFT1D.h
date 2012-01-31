/**
 * @file cxx/sp/sp/FFT1D.h
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using FFTPACK
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

#ifndef BOB_SP_FFT1D_H
#define BOB_SP_FFT1D_H

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
      * @brief This class implements a 1D Discrete Fourier Transform based on 
      * the Netlib FFTPACK library. It is used as a base class for FFT1D and
      * IFFT1D classes.
      */
    class FFT1DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */
        FFT1DAbstract( const int length);

        /**
          * @brief Destructor
          */
        virtual ~FFT1DAbstract();

        /**
          * @brief process an array by applying the FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
          blitz::Array<std::complex<double>,1>& dst) = 0;

        /**
          * @brief Reset the FFT1D object for the given 1D shape
          */
        void reset(const int length);

        /**
          * @brief Get the current height of the FFT1D object
          */
        inline const int getLength() { return m_length; }

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
        int m_length;
    };


    /**
      * @brief This class implements a direct 1D Discrete Fourier Transform 
      * based on the FFTPACK library
      */
    class FFT1D: public FFT1DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        FFT1D( const int length);

        /**
          * @brief process an array by applying the direct FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
          blitz::Array<std::complex<double>,1>& dst);
    };


    /**
      * @brief This class implements a inverse 1D Discrete Fourier Transform 
      * based on the FFTPACK library
      */
    class IFFT1D: public FFT1DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IFFT1D( const int length);

        /**
          * @brief process an array by applying the inverse FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
          blitz::Array<std::complex<double>,1>& dst);
    };

  }
/**
 * @}
 */
}

#endif /* BOB_SP_FFT1D_H */
