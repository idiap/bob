/**
 * @file src/cxx/ip/src/LBP4R.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief LBP4R implementation
 */

#include "ip/LBP4R.h"

namespace ip = Torch::ip;

ip::LBP4R::LBP4R(const double R, const bool circular, const bool to_average,
    const bool add_average_bit, const bool uniform, 
    const bool rotation_invariant): 
  LBP(4,R,circular,to_average,add_average_bit,uniform,rotation_invariant)
{
  // Initialize the lookup tables
  init_lut_RI();
  init_lut_U2();
  init_lut_U2RI();
  init_lut_normal();
  init_lut_add_average_bit();
  init_lut_current();
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
  m_lut_RI(0) = 1;
  // 3 0's + 1 1's
  m_lut_RI(1) = 2;
  m_lut_RI(2) = 2;
  m_lut_RI(4) = 2;
  m_lut_RI(8) = 2;
  // 2 0's + 2 1's
  m_lut_RI(3) = 3;
  m_lut_RI(5) = 3;
  m_lut_RI(6) = 3;
  m_lut_RI(9) = 3;
  m_lut_RI(10) = 3;
  m_lut_RI(12) = 3;
  // 1 0's + 3 1's
  m_lut_RI(7) = 4;
  m_lut_RI(11) = 4;
  m_lut_RI(13) = 4;
  m_lut_RI(14) = 4;
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
  m_lut_U2RI(0) = 0;

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
  m_lut_U2RI(12) = 3;

  // E) Only three adjacent bits are 1 rest are 0's
  m_lut_U2RI(7) = 4;
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
