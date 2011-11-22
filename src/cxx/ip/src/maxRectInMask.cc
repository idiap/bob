/**
 * @file cxx/ip/src/maxRectInMask.cc
 * @date Mon Apr 18 20:25:30 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include "ip/maxRectInMask.h"

namespace ip = Torch::ip;
namespace ipd = Torch::ip::detail;

bool
ipd::isTrue( const blitz::Array<bool,2>& src, int y0, int x0, int y1, int x1)
{
  for(int j=y0; j<=y1; ++j)
    for(int i=x0; i<=x1; ++i)
      if(!src(j,i))
        return false;
  return true;
}

const blitz::TinyVector<int,4> 
ip::maxRectInMask( const blitz::Array<bool,2>& src)
{
  const int height = src.extent(0);
  const int width = src.extent(1);
  blitz::TinyVector<int,4> cur_sol = 0;
  int cur_max_area = 0;

  for( int y0=0; y0<height; ++y0)
  {
    // Find first true value
    int i_min = 0;
    bool found_true = false;
    for(int x0=0; x0<width; ++x0)
      if(src(y0,x0))
      {
        i_min=x0;
        found_true = true;
        break;
      }
    if(!found_true)
      continue;
    
    // Find last true value
    int i_max = width;
    for(int x0=i_min+1; x0<width; ++x0)
      if(!src(y0,x0))
      {
        i_max = x0;
        break;
      }

    // Find largest rectangle visiting all the possible combinations 
    for(int x0=i_min; x0<i_max; ++x0) 
    {
      int cur_i_max = i_max;
      for(int y1=y0; y1<height; ++y1)
        for(int x1=x0; x1<cur_i_max; ++x1)
        {
          if(ipd::isTrue(src,y1,x0,y1,x1))
          {
            int area = (y1-y0+1)*(x1-x0+1);
            if( area > cur_max_area )
            {
              cur_max_area = area;
              cur_sol(0) = y0;
              cur_sol(1) = x0;
              cur_sol(2) = y1-y0+1;
              cur_sol(3) = x1-x0+1;
            }
          }
          else
          {
            if(x1 < cur_i_max)
              cur_i_max = x1;
            break;
          }
        }
    } 
  }

  return cur_sol;
}

