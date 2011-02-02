/**
 * @file src/cxx/sp/sp/FCT.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based Fast Cosine Transform using Lapack functions
 */

#ifndef TORCH5SPRO_SP_FCT_H
#define TORCH5SPRO_SP_FCT_H

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
     * @brief 1D FCT of a 1D blitz array
     */
    blitz::Array<double,1> fct(const blitz::Array<double,1>& A);

    /**
     * @brief 1D inverse FCT of a 1D blitz array
     */
    blitz::Array<double,1> ifct(const blitz::Array<double,1>& A);

    /**
     * @brief 2D FCT of a 2D blitz array
     */
    blitz::Array<double,2> fct(const blitz::Array<double,2>& A);

    /**
     * @brief 2D inverse FCT of a 2D blitz array
     */
    blitz::Array<double,2> ifct(const blitz::Array<double,2>& A);

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_FCT_H */
