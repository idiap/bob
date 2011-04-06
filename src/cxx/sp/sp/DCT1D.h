/**
 * @file src/cxx/sp/sp/DCT1D.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based 1D Fast Cosine Transform using FFTPACK 
 * functions
 */

#ifndef TORCH5SPRO_SP_DCT1D_H
#define TORCH5SPRO_SP_DCT1D_H

#include "core/logging.h"

namespace Torch {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp {

    /**
      * @brief This class implements a 1D Discrete Cosine Transform based on 
      * the Netlib FFTPACK library. It is used as a base class for DCT1D and
      * IDCT1D classes.
      */
    class DCT1DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */
        DCT1DAbstract( const int length);

        /**
          * @brief Destructor
          */
        virtual ~DCT1DAbstract();

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst) = 0;

        /**
          * @brief Reset the DCT1D object for the given 1D shape
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

        /**
          * Normalization factors
          */
        double m_sqrt_1l;
        double m_sqrt_2l;
        double m_sqrt_1byl;
        double m_sqrt_2byl;
    };


    /**
      * @brief This class implements a direct 1D Discrete Cosine Transform 
      * based on the FFTPACK library
      */
    class DCT1D: public DCT1DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT1D( const int length);

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst);
    };


    /**
      * @brief This class implements a inverse 1D Discrete Cosine Transform 
      * based on the FFTPACK library
      */
    class IDCT1D: public DCT1DAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IDCT1D( const int length);

        /**
          * @brief process an array by applying the inverse DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst);
    };

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_DCT1D_H */
