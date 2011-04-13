/**
 * @file src/cxx/sp/sp/DCT1DNaive.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a naive 1D Discrete Cosine Transform
 */

#ifndef TORCH5SPRO_SP_DCT1D_NAIVE_H
#define TORCH5SPRO_SP_DCT1D_NAIVE_H

#include <blitz/array.h>

namespace Torch {
/**
 * \ingroup libsp_api
 * @{
 *
 */
  namespace sp { namespace detail {

    /**
      * @brief This class implements a naive 1D Discrete Cosine Transform.
      */
    class DCT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */
        DCT1DNaiveAbstract( const int length);

        /**
          * @brief Destructor
          */
        virtual ~DCT1DNaiveAbstract();

        /**
          * @brief process an array by applying the DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst) = 0;

        /**
          * @brief Reset the DCT1DNaive object for the given 1D shape
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

      protected:
        /**
          * Private attributes
          */
        int m_length;

        /**
          * Working array
          */
        blitz::Array<double,1> m_wsave; 

        /**
          * Normalization factors
          */
        double m_sqrt_1l;
        double m_sqrt_2l;
    };


    /**
      * @brief This class implements a naive direct 1D Discrete Cosine 
      * Transform
      */
    class DCT1DNaive: public DCT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */ 
        DCT1DNaive( const int length);

        /**
          * @brief process an array by applying the direct DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst);
      
      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<double,1>& src,
          blitz::Array<double,1>& dst);
    };


    /**
      * @brief This class implements a naive inverse 1D Discrete Cosine 
      * Transform 
      */
    class IDCT1DNaive: public DCT1DNaiveAbstract
    {
      public:
        /**
          * @brief Constructor: Initialize working array
          */ 
        IDCT1DNaive( const int length);

        /**
          * @brief process an array by applying the inverse DCT
          */
        virtual void operator()(const blitz::Array<double,1>& src, 
          blitz::Array<double,1>& dst);

      private:
        /**
          * @brief process an array assuming that all the 'check' are done
          */
        void processNoCheck(const blitz::Array<double,1>& src,
          blitz::Array<double,1>& dst);
    };

  }}
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_DCT1D_NAIVE_H */
