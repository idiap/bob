#ifndef TORCH5SPRO_CORE_TENSOR_H
#define TORCH5SPRO_CORE_TENSOR_H

//#include "core/array_common.h"
#include <blitz/array.h>
#include "core/TensorWrapper.h"

/**
 * \defgroup libcore_api libCore API
 * @{
 *
 *  The libCore API.
 */

namespace Torch 
{
  /**
   *  @brief Specify the size of some tensor
   */
  struct TensorSize
  {
    TensorSize()
    {
      n_dimensions = 0;
      size[0] = 0;
      size[1] = 0;
      size[2] = 0;
      size[3] = 0;
    }
    TensorSize(int dim0)
    {
      n_dimensions = 1;
      size[0] = dim0;
      size[1] = 0;
      size[2] = 0;
      size[3] = 0;
    }

    TensorSize(int dim0, int dim1)
    {
      n_dimensions = 2;
      size[0] = dim0;
      size[1] = dim1;
      size[2] = 0;
      size[3] = 0;
    }

    TensorSize(int dim0, int dim1, int dim2)
    {
      n_dimensions = 3;
      size[0] = dim0;
      size[1] = dim1;
      size[2] = dim2;
      size[3] = 0;
    }

    TensorSize(int dim0, int dim1, int dim2, int dim3)
    {
      n_dimensions = 4;
      size[0] = dim0;
      size[1] = dim1;
      size[2] = dim2;
      size[3] = dim3;
    }

    int n_dimensions;
    int size[4];
  };


  /**
   *  @brief TensorRegion: specify some region of a tensor
   *   - [x : x + size] for each dimension
   */
  struct TensorRegion
  {
    TensorRegion()
      : n_dimensions(0)
    {
      pos[0] = 0; size[0] = 0;
      pos[1] = 0; size[1] = 0;
      pos[2] = 0; size[2] = 0;
      pos[3] = 0; size[3] = 0;
    }
    TensorRegion(long x0, long size0)
      : n_dimensions(1)
    {
      pos[0] = x0; size[0] = size0;
      pos[1] = 0; size[1] = 0;
      pos[2] = 0; size[2] = 0;
      pos[3] = 0; size[3] = 0;
    }
    TensorRegion(long x0, long x1, long size0, long size1)
      : n_dimensions(2)
    {
      pos[0] = x0; size[0] = size0;
      pos[1] = x1; size[1] = size1;
      pos[2] = 0; size[2] = 0;
      pos[3] = 0; size[3] = 0;
    }
    TensorRegion(long x0, long x1, long x2, long size0, long size1, long size2)
      : n_dimensions(3)
    {
      pos[0] = x0; size[0] = size0;
      pos[1] = x1; size[1] = size1;
      pos[2] = x2; size[2] = size2;
      pos[3] = 0; size[3] = 0;
    }
    TensorRegion(long x0, long x1, long x2, long x3, long size0, long size1, long size2, long size3)
      : n_dimensions(4)
    {
      pos[0] = x0; size[0] = size0;
      pos[1] = x1; size[1] = size1;
      pos[2] = x2; size[2] = size2;
      pos[3] = x3; size[3] = size3;
    }

    int	n_dimensions;
    long	pos[4];
    long	size[4];
  };


}

/**
 * @}
 */

/**
  @page libCore Core: The Core module of Torch

  @section intro Introduction

  Core contains the core entities of the Torch library.

  @section api Documentation
  - @ref libcore_api "libCore API"

*/

#endif
