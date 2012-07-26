#ifndef BOB_VISIONER_INTEGRAL_H
#define BOB_VISIONER_INTEGRAL_H

#include "visioner/vision/vision.h"

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
