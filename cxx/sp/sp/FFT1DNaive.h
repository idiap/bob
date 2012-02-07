/**
 * @file cxx/sp/sp/FFT1DNaive.h
 * @date Wed Apr 13 23:08:13 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implement a naive 1D Discrete Fourier Transform
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

#ifndef BOB_SP_FFT1D_NAIVE_H
#define BOB_SP_FFT1D_NAIVE_H

#include <complex>
#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp { namespace detail {

    /**
      * @brief This class implements a naive 1D Discrete Fourier Transform.
      */
    class FFT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */
        FFT1DNaiveAbstract( const int length);

        /**
          * @brief Destructor
          */
        virtual ~FFT1DNaiveAbstract();

        /**
          * @brief process an array by applying the FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
          blitz::Array<std::complex<double>,1>& dst) = 0;

        /**
          * @brief Reset the FFT1DNaive object for the given 1D shape
          */
        void reset(const int length);

        /**
          * @brief Get the current height of the FFT1D object
          */
        inline const int getLength() { return m_length; }

      private:
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
        blitz::Array<std::complex<double>,1> m_wsave; 
    };


    /**
      * @brief This class implements a naive direct 1D Discrete Fourier 
      * Transform
      */
    class FFT1DNaive: public FFT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        FFT1DNaive( const int length);

        /**
          * @brief process an array by applying the direct FFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
          blitz::Array<std::complex<double>,1>& dst);
      
      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
          blitz::Array<std::complex<double>,1>& dst);
    };


    /**
      * @brief This class implements a naive inverse 1D Discrete Fourier
      * Transform 
      */
    class IFFT1DNaive: public FFT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IFFT1DNaive( const int length);

        /**
          * @brief process an array by applying the inverse DFT
          */
        virtual void operator()(const blitz::Array<std::complex<double>,1>& src, 
          blitz::Array<std::complex<double>,1>& dst);

      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<std::complex<double>,1>& src,
          blitz::Array<std::complex<double>,1>& dst);
    };

  }}
/**
 * @}
 */
}

#endif /* BOB_SP_FFT1D_NAIVE_H */
