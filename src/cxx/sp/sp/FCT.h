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
    void fct(const blitz::Array<double,1>& A, blitz::Array<double,1>& B);

    /**
     * @brief 1D inverse FCT of a 1D blitz array
     */
    void ifct(const blitz::Array<double,1>& A, blitz::Array<double,1>& B);

    /**
     * @brief 2D FCT of a 2D blitz array
     */
    void fct(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);

    /**
     * @brief 2D inverse FCT of a 2D blitz array
     */
    void ifct(const blitz::Array<double,2>& A, blitz::Array<double,2>& B);

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_FCT_H */
