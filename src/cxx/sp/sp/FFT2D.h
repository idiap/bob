/**
 * @file src/cxx/sp/sp/FFT2D.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 2D Fast Fourier Transform using FFTPACK 
 * functions
 */

#ifndef TORCH5SPRO_SP_FFT2D_H
#define TORCH5SPRO_SP_FFT2D_H

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
          * @brief Initialize the working arrays
          */
        void initWorkingArrays();

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
          * Working arrays
          */
        double *m_wsave_w; 
        double *m_wsave_h; 
        std::complex<double> *m_col_tmp;
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

#endif /* TORCH5SPRO_SP_FFT2D_H */
