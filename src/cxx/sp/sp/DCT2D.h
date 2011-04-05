/**
 * @file src/cxx/sp/sp/DCT2D.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based Fast Cosine Transform using FFTPACK functions
 */

#ifndef TORCH5SPRO_SP_DCT2D_H
#define TORCH5SPRO_SP_DCT2D_H

#include "core/array_common.h"
#include "core/logging.h"
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
      * Netlib FFTPACK library
      */
    class DCT2D
    {
      public:
        /**
          * @brief Constructor: Initialize working arrays
          */
        DCT2D( const int height, const int width);

        /**
          * @brief Destructor
          */
        ~DCT2D();

        /**
          * @brief process an array by applying the DCT
          */
        void operator()(const blitz::Array<double,2>& src, 
          blitz::Array<double,2>& dst);

      private:
        /**
          * @brief Initialize the working arrays
          */
        void initWorkingArrays();

        /**
          * @brief Deallocate memory
          */
        void cleanup();

        /**
          * Private attributes
          */
        int m_height;
        int m_width;
        double *m_wsave_w; 
        double *m_wsave_h; 
        double *m_col_tmp;

        double m_sqrt_1h;
        double m_sqrt_2h;
        double m_sqrt_1w;
        double m_sqrt_2w;
    };
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_DCT2D_H */
