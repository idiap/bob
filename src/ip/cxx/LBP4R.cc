/**
 * @file cxx/ip/src/LBP4R.cc
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief LBP4R implementation
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

#include <boost/make_shared.hpp>
#include "bob/ip/LBP4R.h"

namespace ip = bob::ip;

ip::LBP4R::LBP4R(const double R,
    const bool circular,
    const bool to_average,
    const bool add_average_bit,
    const bool uniform, 
    const bool rotation_invariant,
    const int eLBP_type): 
  ip::LBP(4,R,R,circular,to_average,add_average_bit,uniform,rotation_invariant,
      eLBP_type)
{
  init_luts();
}

ip::LBP4R::LBP4R(const double R,
    const double R2,
    const bool circular,
    const bool to_average,
    const bool add_average_bit,
    const bool uniform, 
    const bool rotation_invariant,
    const int eLBP_type): 
  ip::LBP(4,R,R2,circular,to_average,add_average_bit,uniform,rotation_invariant,
      eLBP_type)
{
  init_luts();
}


ip::LBP4R::LBP4R(const ip::LBP4R& other): 
  ip::LBP(other)
{
  init_luts();
}

ip::LBP4R::~LBP4R() { }

ip::LBP4R& ip::LBP4R::operator= (const ip::LBP4R& other) {
  ip::LBP::operator=(other);
  return *this;
}

boost::shared_ptr<ip::LBP> ip::LBP4R::clone() const {
  return boost::make_shared<ip::LBP4R>(*this);
}

int ip::LBP4R::getMaxLabel() const
{
  return  (m_rotation_invariant ? 6 :
            (m_uniform ? 15 : // Uniform
              (m_to_average && m_add_average_bit ? // i.e. 2^5=32 vs. 2^4=16
                32 : 16)        // i.e. 2^4=16)
             )
          ); 
}

void ip::LBP4R::init_lut_RI()
{
  m_lut_RI.resize(16);
  // all 0's
  m_lut_RI(0) = 0;
  // binary pattern 0001
  m_lut_RI(1) = 1;
  m_lut_RI(2) = 1;
  m_lut_RI(4) = 1;
  m_lut_RI(8) = 1;
  // binary pattern 0011
  m_lut_RI(3) = 2;
  m_lut_RI(6) = 2;
  m_lut_RI(12) = 2;
  m_lut_RI(9) = 2;
  // binary pattern 0101
  m_lut_RI(5) = 3;
  m_lut_RI(10) = 3;
  // binary pattern 0111
  m_lut_RI(7) = 4;
  m_lut_RI(14) = 4;
  m_lut_RI(11) = 4;
  m_lut_RI(13) = 4;
  // all 1's
  m_lut_RI(15) = 5;
}


void ip::LBP4R::init_lut_U2()
{
  m_lut_U2.resize(16);
  // A) All non uniform patterns have a label of 0.
  m_lut_U2 = 0;

  // B) LBP pattern with 0 bit to 1
  m_lut_U2(0) = 1;

  // C) LBP patterns with 1 bit to 1
  m_lut_U2(8) = 2;
  m_lut_U2(4) = 3;
  m_lut_U2(2) = 4;
  m_lut_U2(1) = 5;

  // D) LBP patterns with 2 bits to 1
  m_lut_U2(8+4) = 6;
  m_lut_U2(4+2) = 7;
  m_lut_U2(2+1) = 8;
  m_lut_U2(1+8) = 9;

  // E) LBP patterns with 3 bits to 1
  m_lut_U2(8+4+2) = 10;
  m_lut_U2(4+2+1) = 11;
  m_lut_U2(2+1+8) = 12;
  m_lut_U2(1+8+4) = 13;

  // F) LBP patterns with 4 bits to 1
  m_lut_U2(8+4+2+1) = 14;
}

void ip::LBP4R::init_lut_U2RI()
{
  m_lut_U2RI.resize(16);
  // A) All non uniform patterns have a label of 0.
  m_lut_U2RI = 0;

  // B) All bits are 0
  m_lut_U2RI(0) = 1;

  // C) Only one bit is 1 rest are 0's
  m_lut_U2RI(1) = 2;
  m_lut_U2RI(2) = 2;
  m_lut_U2RI(4) = 2;
  m_lut_U2RI(8) = 2;

  // D) Only two adjacent bits are 1 rest are 0's
  m_lut_U2RI(3) = 3;
  m_lut_U2RI(6) = 3;
  m_lut_U2RI(9) = 3;
  m_lut_U2RI(12) = 3;

  // E) Only three adjacent bits are 1 rest are 0's
  m_lut_U2RI(7) = 4;
  m_lut_U2RI(11) = 4;
  m_lut_U2RI(13) = 4;
  m_lut_U2RI(14) = 4;

  // F) four adjacent bits are 1
  m_lut_U2RI(15) = 5;
}

void ip::LBP4R::init_lut_add_average_bit()
{
  m_lut_add_average_bit.resize(32);
  blitz::firstIndex i;
  m_lut_add_average_bit = i;
}

void ip::LBP4R::init_lut_normal()
{
  m_lut_normal.resize(16);
  blitz::firstIndex i;
  m_lut_normal = i;
}
