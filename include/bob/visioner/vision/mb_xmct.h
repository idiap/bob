/**
 * @file visioner/visioner/vision/mb_xmct.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_VISIONER_MB_XMCT_H
#define BOB_VISIONER_MB_XMCT_H

#include "visioner/util/matrix.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the <NCELLSX> x <NCELLSY> multi-block MCT codes:
  //	- for all possible locations,
  //	- at a given top-left location.
  // NB. (ii) is the integral of the 2D signal.
  // NB. (cx, cy) is the cell size (the scale of the code). 
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename TII, int NCELLSX, int NCELLSY, typename TCODE>
    bool mb_dense_mct(const Matrix<TII>& ii, int cx, int cy, Matrix<TCODE>& codes)
    {
      if (ii.empty())
      {
        return false;
      }

      const int w = ii.cols(), h = ii.rows();
      const int min_x = 0, max_x = w - NCELLSX * cx;
      const int min_y = 0, max_y = h - NCELLSY * cy;                
      const int odx = cx * NCELLSX / 2, ody = cy * NCELLSY / 2;
      const int mb_w = NCELLSX * cx, mb_h = NCELLSY * cy;

      codes.resize(h, w);
      codes.fill(0);

      for (int y = min_y; y < max_y; y ++)
      {
        for (int x = min_x; x < max_x; x ++)
        {
          const TII avg = 
            (ii(y, x) + ii(y + mb_h, x + mb_w) - ii(y + mb_h, x) - ii(y, x + mb_w)) / 
            (NCELLSX * NCELLSY);

          TCODE code = 0;                                
          for (int icy = 0, dy = y, bit = 0; icy < NCELLSY; icy ++, dy += cy)
          {
            for (int icx = 0, dx = x; icx < NCELLSX; icx ++, dx += cx, bit ++)
            {
              const TII val = 
                ii(dy, dx) + ii(dy + cy, dx + cx) - 
                ii(dy, dx + cx) - ii(dy + cy, dx);
              code |= (val > avg) << bit;
            }
          }

          codes(y + ody, x + odx) = code;
        }
      }

      // OK
      return true;
    }        

  template <typename TII, int NCELLSX, int NCELLSY, typename TCODE>
    TCODE mb_mct(const Matrix<TII>& ii, int x, int y, int cx, int cy)
    {
      const int mb_w = NCELLSX * cx, mb_h = NCELLSY * cy;		
      const TII avg = 
        (ii(y, x) + ii(y + mb_h, x + mb_w) - ii(y + mb_h, x) - ii(y, x + mb_w)) / 
        (NCELLSX * NCELLSY);

      TCODE code = 0;                                
      for (int icy = 0, dy = y, bit = 0; icy < NCELLSY; icy ++, dy += cy)
      {
        for (int icx = 0, dx = x; icx < NCELLSX; icx ++, dx += cx, bit ++)
        {
          const TII val = 
            ii(dy, dx) + ii(dy + cy, dx + cx) - 
            ii(dy, dx + cx) - ii(dy + cy, dx);
          code |= (val > avg) << bit;
        }
      }

      return code;
    }

}}

#endif // BOB_VISIONER_MB_XMCT_H
