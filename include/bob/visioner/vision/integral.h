/**
 * @file bob/visioner/vision/integral.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_INTEGRAL_H
#define BOB_VISIONER_INTEGRAL_H

#include "bob/visioner/vision/vision.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the integral image for some matrix
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename TIn, typename TOut>
    void integral(const Matrix<TIn>& in, Matrix<TOut>& out)
    {
      const int w = in.cols(), h = in.rows();
      if (w < 2 || h < 2)
      {
        return;
      }
      out.resize(h, w);

      const TIn* src = &in(0, 0);
      TOut* crt_row = &out(0, 0);
      TOut* last_row = &out(0, 0);

      // Initialize the first row
      crt_row[0] = *(src ++);
      for (int x = 1; x < w; x ++)
      {
        crt_row[x] = crt_row[x - 1] + (*(src ++));
      }
      crt_row = &out(1, 0);

      // Each row is computed using the previous one
      for (int y = 1; y < h; y ++)
      {
        TOut row_sum = (TOut)0;

        for (int x = 0; x < w; x ++)
        {
          row_sum += *(src ++);
          *(crt_row ++) = *(last_row ++) + row_sum;
        }
      }
    }

}}

#endif // BOB_VISIONER_INTEGRAL_H
