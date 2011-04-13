/**
 * @file src/cxx/sp/sp/DCT2D.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 2D Fast Cosine Transform using FFTPACK 
 * functions
 */

#ifndef TORCH5SPRO_SP_DCT2D_H
#define TORCH5SPRO_SP_DCT2D_H

#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp {

    /**
      * @brief This class implements a Discrete Cosine Transform based on the
      * Netlib FFTPACK library. It is used as a base class for DCT2D and 
      * IDCT2D classes.
      */
    class DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        DCT2DAbstract( const int height, const int width);

        /**
          * @brief Destructor
          */
        virtual ~DCT2DAbstract();

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst) = 0;

        /**
          * @brief Reset the DCT2D object for the given 2D shape
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
        double *m_col_tmp;

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
      * based on the FFTPACK library
      */
    class DCT2D: public DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT2D( const int height, const int width);

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);
    };


    /**
      * @brief This class implements a inverse 2D Discrete Cosine Transform 
      * based on the FFTPACK library
      */
    class IDCT2D: public DCT2DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        IDCT2D( const int height, const int width);

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

#endif /* TORCH5SPRO_SP_DCT2D_H */
