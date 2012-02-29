/**
 * @file cxx/ip/src/LBP16R.cc
 * @date Mon Feb 27 18:24:26 CET 2012
 * @author Ivana Chingovska <ivana.chingovska@idiap.ch>
 *
 * @brief LBP8R implementation
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

#include "ip/LBP16R.h"

namespace ip = bob::ip;

ip::LBP16R::LBP16R(const double R, const bool circular, const bool to_average,
    const bool add_average_bit, const bool uniform, 
    const bool rotation_invariant): 
  LBP(16,R,circular,to_average,add_average_bit,uniform,rotation_invariant)
{
  // Initialize the lookup tables
  init_luts();
}

int ip::LBP16R::getMaxLabel() const
{
return  (m_rotation_invariant ?
            (m_uniform ? 18 : // Rotation invariant + uniform
                         4116) // Rotation invariant
          : (m_uniform ? 243 : // Uniform
              (m_to_average && m_add_average_bit ? 131072 : // i.e. 2^17=512 vs. 2^16=65536
                                            65536)       // i.e. 2^8=256)
            )
        );
}

void ip::LBP16R::init_lut_RI()
{
  m_lut_RI.resize(65536);
  
  // just find the minimum of all the rotations of one binary pattern
  for (int i=0; i<65536; i++)
  {
    int min = i;
    for (int j=0; j<16; j++)
    {
      int k = right_shift_circular(i, j+1, 16); // rotate shift for the j+1 bit
      if (k<min) min=k;
    }
    m_lut_RI(i) = min;
  }
}


void ip::LBP16R::init_lut_U2()
{
  m_lut_U2.resize(65536);
  int counter = 0 ;
  int bases[] = {1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
	// all non uniform patterns have a label of 0.
  m_lut_U2 = 0; counter++;

	// LBP pattern with 0 bit to 1
	m_lut_U2(0) = 1; counter++;

	// all the other LBP patterns with bases[i] next-to-each-other bits equal to 1
  for (int i=0; i < 16; i++)
  {
    for (int j=bases[i];;)
    {
      m_lut_U2(j) = counter;
      counter++;
      j = right_shift_circular(j,1,16);
      if (j == bases[i]) break;
    }
  }
}

void ip::LBP16R::init_lut_U2RI()
{
  m_lut_U2RI.resize(65536);
  int counter = 0;
  int bases[] = {1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535};
	// all non uniform patterns have a label of 0.
  m_lut_U2RI = 0; counter++;

  // LBP pattern with zero bits equal to 1
	m_lut_U2RI(0) = 1; counter++;

  // all the other LBP patterns with bases[i] next-to-each-other bits equal to 1
  for (int i=0; i < 16; i++)
  {
    for (int j=bases[i];;)
    {
      m_lut_U2RI(j) = counter;
      j = right_shift_circular(j,1,16);
      if (j == bases[i]) break;
    }
    counter++;
  }
}


void ip::LBP16R::init_lut_add_average_bit()
{
  m_lut_add_average_bit.resize(11072);
  blitz::firstIndex i;
  m_lut_add_average_bit = i;
}

void ip::LBP16R::init_lut_normal()
{
  m_lut_normal.resize(65536);
  blitz::firstIndex i;
  m_lut_normal = i;
}
