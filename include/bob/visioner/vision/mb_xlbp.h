/**
 * @file bob/visioner/vision/mb_xlbp.h
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

#ifndef BOB_VISIONER_MB_XLBP_H
#define BOB_VISIONER_MB_XLBP_H

#include "bob/visioner/util/matrix.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the multi-block (+modified/t-transitional/d-directional ...) 8-bit LBP codes.
  // NB. (ii) is the integral of the 2D signal.
  // NB. (cx, cy) is the cell size (the scale of the code). 
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename TII, typename TCODE>
    TCODE mb_8bit_code_lt(TII v1, TII v2, TII v3, TII v4, TII v5, TII v6, TII v7, TII v8, TII cmp)
    {
      return  ((v1 < cmp) << 7) |
        ((v2 < cmp) << 6) |
        ((v3 < cmp) << 5) |
        ((v4 < cmp) << 4) |
        ((v5 < cmp) << 3) |
        ((v6 < cmp) << 2) |
        ((v7 < cmp) << 1) |
        ((v8 < cmp) << 0);
    }

  template <typename TII, typename TCODE>
    TCODE mb_8bit_code_gt(TII v1, TII v2, TII v3, TII v4, TII v5, TII v6, TII v7, TII v8, TII cmp)
    {
      return  ((v1 > cmp) << 7) |
        ((v2 > cmp) << 6) |
        ((v3 > cmp) << 5) |
        ((v4 > cmp) << 4) |
        ((v5 > cmp) << 3) |
        ((v6 > cmp) << 2) |
        ((v7 > cmp) << 1) |
        ((v8 > cmp) << 0);
    }

  template <typename TII, typename TCODE>
    TCODE tlbp(TII p1, TII p2, TII p3, TII p4, TII p5, TII p6, TII p7, TII p8)
    {
      return  ((p1 > p2) << 7) |
        ((p2 > p3) << 6) |
        ((p3 > p4) << 5) |
        ((p4 > p5) << 4) |
        ((p5 > p6) << 3) |
        ((p6 > p7) << 2) |
        ((p7 > p8) << 1) |
        ((p8 > p1) << 0);
    }

  ////////////////////////////////////////////////////////////////////////////
  //      P00     P01     P02     P03
  //      P10     P11     P12     P13
  //      P20     P21     P22     P23
  //      P30     P31     P32     P33
  ////////////////////////////////////////////////////////////////////////////
  //      p1      p2      p3
  //      p8      xx      p4
  //      p7      p6      p5      
  ////////////////////////////////////////////////////////////////////////////
#define INIT_xLBP \
  const int x0 = x, x1 = x0 + cx, x2 = x1 + cx, x3 = x2 + cx;\
  const int y0 = y, y1 = y0 + cy, y2 = y1 + cy, y3 = y2 + cy;\
  \
  const TII P00 = ii(y0, x0), P01 = ii(y0, x1), P02 = ii(y0, x2), P03 = ii(y0, x3);\
  const TII P10 = ii(y1, x0), P11 = ii(y1, x1), P12 = ii(y1, x2), P13 = ii(y1, x3);\
  const TII P20 = ii(y2, x0), P21 = ii(y2, x1), P22 = ii(y2, x2), P23 = ii(y2, x3);\
  const TII P30 = ii(y3, x0), P31 = ii(y3, x1), P32 = ii(y3, x2), P33 = ii(y3, x3);\
  \
  const TII p1 = P00 + P11 - P01 - P10;\
  const TII p2 = P01 + P12 - P02 - P11;\
  const TII p3 = P02 + P13 - P03 - P12;\
  const TII p4 = P12 + P23 - P13 - P22;\
  const TII p5 = P22 + P33 - P23 - P32;\
  const TII p6 = P21 + P32 - P22 - P31;\
  const TII p7 = P20 + P31 - P21 - P30;\
  const TII p8 = P10 + P21 - P11 - P20;     

  template <typename TII, typename TCODE>
    TCODE mb_lbp(const Matrix<TII>& ii, int x, int y, int cx, int cy)
    {
      INIT_xLBP

        const TII pc = P11 + P22 - P12 - P21;

      return mb_8bit_code_gt<TII, TCODE>(p1, p2, p3, p4, p5, p6, p7, p8, pc);
    }

  template <typename TII, typename TCODE>
    TCODE mb_tlbp(const Matrix<TII>& ii, int x, int y, int cx, int cy)
    {
      INIT_xLBP

        return tlbp<TII, TCODE>(p1, p2, p3, p4, p5, p6, p7, p8);
    }

  template <typename TII, typename TCODE>
    TCODE mb_dlbp(const Matrix<TII>& ii, int x, int y, int cx, int cy)
    {
      INIT_xLBP

        const TII pc = P11 + P22 - P12 - P21;

      const TII p1c = p1 - pc;
      const TII p2c = p2 - pc;
      const TII p3c = p3 - pc;
      const TII p4c = p4 - pc;                
      const TII p5c = p5 - pc;
      const TII p6c = p6 - pc;
      const TII p7c = p7 - pc;
      const TII p8c = p8 - pc;

      return  ((p1c * p5c > 0) << 7) |        
        ((p2c * p6c > 0) << 5) |        
        ((p3c * p7c > 0) << 3) |        
        ((p4c * p8c > 0) << 1) |        

        ((abs(p1c) > abs(p5c)) << 6) |
        ((abs(p2c) > abs(p6c)) << 4) |
        ((abs(p3c) > abs(p7c)) << 2) |
        ((abs(p4c) > abs(p8c)) << 0);
    }        

  template <typename TII, typename TCODE>
    TCODE mb_mlbp(const Matrix<TII>& ii, int x, int y, int cx, int cy)
    {
      INIT_xLBP

        const TII avg = (P00 + P33 - P03 - P30) / 9;

      return mb_8bit_code_gt<TII, TCODE>(p1, p2, p3, p4, p5, p6, p7, p8, avg);
    }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Compute the dense MB-xLBP feature maps.
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename TII, typename TCODE,
           int NCELLSX, int NCELLSY,
           TCODE (*TOP) (const Matrix<TII>& ii, int x, int y, int cx, int cy)>
             void mb_dense(const Matrix<TII>& ii, int cx, int cy, Matrix<TCODE>& codes)
             {
               const int w = ii.cols(), h = ii.rows();
               const int min_x = 0, max_x = w - NCELLSX * cx;
               const int min_y = 0, max_y = h - NCELLSY * cy;                
               const int odx = cx * NCELLSX / 2, ody = cy * NCELLSY / 2;

               codes.resize(h, w);
               codes.fill(0);

               for (int y = min_y; y < max_y; y ++)
               {
                 for (int x = min_x; x < max_x; x ++)
                 {
                   codes(y + ody, x + odx) = TOP(ii, x, y, cx, cy);;
                 }
               }
             }

}}

#endif // BOB_VISIONER_MB_XLBP_H
