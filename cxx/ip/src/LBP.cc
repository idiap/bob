/**
 * @file cxx/ip/src/LBP.cc
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief LBP implementation
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#include "ip/LBP.h"

namespace ip = bob::ip;

ip::LBP::LBP(const int P, const double R, const bool circular,
    const bool to_average,const bool add_average_bit, const bool uniform,
    const bool rotation_invariant):
  m_P(P), m_R(R), m_circular(circular), m_to_average(to_average), 
  m_add_average_bit(add_average_bit), m_uniform(uniform), 
  m_rotation_invariant(rotation_invariant),
  m_lut_RI(0), m_lut_U2(0), m_lut_U2RI(0), m_lut_add_average_bit(0), 
  m_lut_normal(0), m_lut_current(0)
{
  updateR(); 
}

void ip::LBP::init_lut_current()
{
  // Reference to the current lookup table
  if(m_rotation_invariant)
  {
    if(m_uniform)
      m_lut_current.reference( m_lut_U2RI );
    else
      m_lut_current.reference( m_lut_RI );
  }
  else
  {
    if(m_uniform)
      m_lut_current.reference( m_lut_U2 );
    else
    {
      if(m_add_average_bit && m_to_average)
        m_lut_current.reference( m_lut_add_average_bit );
      else
        m_lut_current.reference( m_lut_normal );
    }   
  }
}

void ip::LBP::init_luts()
{
  // Initialize the lookup tables
  init_lut_RI();
  init_lut_U2();
  init_lut_U2RI();
  init_lut_normal();
  init_lut_add_average_bit();
  init_lut_current();
}
