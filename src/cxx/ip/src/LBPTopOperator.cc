/**
 * @file cxx/ip/src/LBPTopOperator.cc
 * @date Tue Apr 26 19:20:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "ip/LBPTopOperator.h"
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include "ip/Exception.h"

namespace ip = Torch::ip;

/**
 * A little helper to create the LBP operators in an homogene way.
 */
static ip::LBP* make_lbp(int radius, int points)
{
  ip::LBP* retval = 0;
  if (points != 4 && points != 8) {
    // TODO
    throw Torch::ip::Exception();
  /*  Torch::error("Cannot create %d-point LBP operator (use 4 or 8 only)!",
        points);*/
  }
  else {
    if (points == 4) 
      retval = new ip::LBP4R(radius, false, false, false, true, true);
    else  
      retval = new ip::LBP8R(radius, false, false, false, true, true);
  }
  return retval;
}

ip::LBPTopOperator::LBPTopOperator(int radius_xy, 
                                   int points_xy, 
                                   int radius_xt, 
                                   int points_xt, 
                                   int radius_yt,
                                   int points_yt)
: m_radius_xy(radius_xy),
  m_points_xy(points_xy),
  m_radius_xt(radius_xt),
  m_points_xt(points_xt),
  m_radius_yt(radius_yt),
  m_points_yt(points_yt)
{
  m_lbp_xy = make_lbp(m_radius_xy, m_points_xy);
  m_lbp_xt = make_lbp(m_radius_xt, m_points_xt);
  m_lbp_xy = make_lbp(m_radius_yt, m_points_yt);
}

ip::LBPTopOperator::~LBPTopOperator() {
  delete m_lbp_xy;
  m_lbp_xy = 0;
  delete m_lbp_xt;
  m_lbp_xt = 0;
  delete m_lbp_yt;
  m_lbp_yt = 0;
}
