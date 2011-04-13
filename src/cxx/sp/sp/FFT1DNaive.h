/**
 * @file src/cxx/sp/sp/FFT1DNaive.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a naive 1D Discrete Fourier Transform
 */

#ifndef TORCH5SPRO_SP_FFT1D_NAIVE_H
#define TORCH5SPRO_SP_FFT1D_NAIVE_H

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

#endif /* TORCH5SPRO_SP_FFT1D_NAIVE_H */
