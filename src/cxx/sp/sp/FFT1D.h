/**
 * @file src/cxx/sp/sp/FFT1D.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 1D Fast Fourier Transform using FFTPACK 
 * functions
 */

#ifndef TORCH5SPRO_SP_FFT1D_H
#define TORCH5SPRO_SP_FFT1D_H

#include <complex>
#include <blitz/array.h>

namespace Torch {
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
          * @brief Initialize the working array
          */
        void initWorkingArray();

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

        /**
          * Working array
          */
        double *m_wsave; 
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

#endif /* TORCH5SPRO_SP_FFT1D_H */
