/**
 * @file cxx/ip/src/LBPTopOperator.cc
 * @date Tue Apr 26 19:20:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief
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

#include "ip/LBPTopOperator.h"
#include "ip/LBP.h"
#include "ip/LBP4R.h"
#include "ip/LBP8R.h"
#include "ip/Exception.h"

namespace ip = bob::ip;

ip::LBPTop::LBPTop(boost::shared_ptr<bob::ip::LBP> lbp_xy, 
			           boost::shared_ptr<bob::ip::LBP> lbp_xt, 
			           boost::shared_ptr<bob::ip::LBP> lbp_yt)
{
  m_lbp_xy = lbp_xy;
  m_lbp_xt = lbp_xt;
  m_lbp_yt = lbp_yt;
}




