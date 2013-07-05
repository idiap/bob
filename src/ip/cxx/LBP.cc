/**
 * @file ip/cxx/LBP.cc
 * @date Wed Apr 20 20:21:19 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Rewritten:
 * @date Wed Apr 10 17:39:21 CEST 2013
 * @author Manuel Günther <manuel.guenther@idiap.ch>
 *
 * @brief LBP implementation
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <bob/ip/LBP.h>

#include <boost/math/constants/constants.hpp>

bob::ip::LBP::LBP(const int P, const double R_y, const double R_x , const bool circular,
    const bool to_average, const bool add_average_bit, const bool uniform,
    const bool rotation_invariant, const bob::ip::ELBPType eLBP_type):
  m_P(P),
  m_R_y(R_y),
  m_R_x(R_x),
  m_circular(circular),
  m_to_average(to_average),
  m_add_average_bit(add_average_bit),
  m_uniform(uniform),
  m_rotation_invariant(rotation_invariant),
  m_eLBP_type(eLBP_type),
  m_lut(0),
  m_positions(0,0)
{
  // sanity check
  if (m_eLBP_type == ELBP_DIRECTION_CODED && m_P%2) {
    throw std::runtime_error("Direction coded LBP types require an even number of neighbors.");
  }
  init();
}

bob::ip::LBP::LBP(const int P, const double R, const bool circular,
    const bool to_average, const bool add_average_bit, const bool uniform,
    const bool rotation_invariant, const bob::ip::ELBPType eLBP_type):
  m_P(P),
  m_R_y(R),
  m_R_x(R),
  m_circular(circular),
  m_to_average(to_average),
  m_add_average_bit(add_average_bit),
  m_uniform(uniform),
  m_rotation_invariant(rotation_invariant),
  m_eLBP_type(eLBP_type),
  m_lut(0),
  m_positions(0,0)
{
  // sanity check
  if (m_eLBP_type == ELBP_DIRECTION_CODED && m_P%2) {
    throw std::runtime_error("Direction coded LBP types require an even number of neighbors.");
  }
  init();
}


bob::ip::LBP::LBP(const bob::ip::LBP& other):
  m_P(other.m_P),
  m_R_y(other.m_R_y),
  m_R_x(other.m_R_x),
  m_circular(other.m_circular),
  m_to_average(other.m_to_average),
  m_add_average_bit(other.m_add_average_bit),
  m_uniform(other.m_uniform),
  m_rotation_invariant(other.m_rotation_invariant),
  m_eLBP_type(other.m_eLBP_type),
  m_lut(0),
  m_positions(0,0)
{
  // sanity check
  if (m_eLBP_type == ELBP_DIRECTION_CODED && m_P%2) {
    throw std::runtime_error("Direction coded LBP types require an even number of neighbors.");
  }
  init();
}

bob::ip::LBP::~LBP() { }

bob::ip::LBP& bob::ip::LBP::operator=(const bob::ip::LBP& other) {
  m_P = other.m_P;
  m_R_y = other.m_R_y;
  m_R_x = other.m_R_x;
  m_circular = other.m_circular;
  m_to_average = other.m_to_average;
  m_add_average_bit = other.m_add_average_bit;
  m_uniform = other.m_uniform;
  m_rotation_invariant = other.m_rotation_invariant;
  m_eLBP_type = other.m_eLBP_type;
  init();
  return *this;
}

uint16_t bob::ip::LBP::right_shift_circular(uint16_t pattern, int spaces)
{
  return (pattern >> spaces | pattern << (m_P-spaces)) & ((1 << m_P) - 1);
}

void bob::ip::LBP::init()
{
  if (m_P < 4)
    throw std::runtime_error("LBP16 codes with less than 4 bits are not supported.");
  // check that the parameters are something useful, what we can handle
  if (m_P != 4 && m_P != 8 && m_P != 16 &&
      (m_uniform || m_rotation_invariant || (m_add_average_bit && m_to_average)))
    throw std::runtime_error("Special LBP types are only implemented for 4, 8, or 16 neighbors.");
  if (m_P == 16 && m_add_average_bit && m_to_average){
    throw std::runtime_error("LBP16 codes with average bit require 17 bits, but our representation is UINT16.");
  }
  if (m_P > 16){
    throw std::runtime_error("LBP codes with more than 16 neighbors are not supported since our representation is UINT16.");
  }

  // initialize the positions
  m_positions.resize(m_P,2);
  if (m_circular){
    double PI = boost::math::constants::pi<double>();
    // compute angle offset since LBP codes do not start at the x axis
    double angle_offset = m_P == 4 ? - 0.5 * PI : - 0.75 * PI;
    for (int p = 0; p < m_P; ++p){
      double angle = angle_offset + 2. * PI * p / m_P;
      m_positions(p,0) = m_R_y * sin(angle);
      m_positions(p,1) = m_R_x * cos(angle);
    }
  }else{ // circular
    blitz::TinyVector<int, 16> d_y, d_x;
    int r_y = (int)round(m_R_y), r_x = (int)round(m_R_x);
    switch (m_P){
      case 4:{
        // 4 neighbors: (-y,0), (0,x), (y,0), (0,-x)
        d_y = -r_y, 0, r_y, 0;
        d_x = 0, r_x, 0, -r_x;
      }break;
      case 8:{
        // 8 neighbors: (-y,-x), (-y,0), (-y,x), (0,x), (y,x), (y,0), (y,-x), (0,-x)
        d_y = -r_y, -r_y, -r_y, 0, r_y, r_y, r_y, 0;
        d_x = -r_x, 0, r_x, r_x, r_x, 0, -r_x, -r_x;
        break;
      }
      case 16:
        // 16 neighbors: ...
        throw std::runtime_error("Rectangular LBP16 codes are not yet implemented.");
      default:
        // any other number of neighbors is not supported
        throw std::runtime_error("Rectangular LBP's with other than 4 and 8 neighbors are not supported.");
    }
    // fill the positions
    for (int p = 0; p < m_P; ++p){
      m_positions(p,0) = d_y[p];
      m_positions(p,1) = d_x[p];
    }
  }

  // initialize the look up table for the current setup
  // initialize all values with 0
  m_lut.resize(1 << m_P);
  m_lut = 0;
  uint16_t lbp_code = 0;
  if (m_uniform){
    // pre-compute uniform pattern bases (i.e., which are rotated to build all uniform patterns)
    std::vector<uint16_t> uniform_pattern_bases(m_P+1);
    for (int p = 1; p < m_P+1; ++p){
      // the pattern generation is adapted to be identical to the old LBP8R uniform pattern generator
      uniform_pattern_bases[p] = ((1 << p) -1) << (m_P - p);
    }
    // all non uniform patterns have a label of 0.
    m_lut = lbp_code++;
    // LBP pattern with all zero bits equal to 1
    m_lut(0) = lbp_code++;
    // compute patterns
    // all the other LBP patterns with bases[i] next-to-each-other bits equal to 1
    for (int p = 1; p < m_P; ++p){
      // assign all shifted versions of the base pattern the same lbp code
      for (int shift = 0; shift < m_P; ++shift){
        int shifted_pattern = right_shift_circular(uniform_pattern_bases[p], shift);
        // assign the shifted pattern
        m_lut(shifted_pattern) = lbp_code;
        if (!m_rotation_invariant)
          // change lbp code for each shift of each pattern
          ++lbp_code;
      }
      if (m_rotation_invariant)
        // change lbp code for each pattern
        ++lbp_code;
    }
    //LBP pattern with all unit bits gets the last code
    m_lut(uniform_pattern_bases[m_P]) = lbp_code;
  }else{ // not uniform
    if (m_rotation_invariant){
      // rotation invariant patterns are not that easy...
      // first, collect all possible RI patterns and assign all patterns to them
      std::vector<bool> found_pattern(1 << m_P);
      std::fill(found_pattern.begin(), found_pattern.end(), false);

      for (int c = 0; c < (1 << m_P); ++c){
        uint16_t pattern = static_cast<uint16_t>(c);
        // search for the LBP code with the smallest integral value
        bool this_pattern_is_new = false;
        for (int p = 0; p < m_P; ++p)
        {
          // generate shifted version of the code
          uint16_t shifted_code = right_shift_circular(pattern, p);
          if (!found_pattern[shifted_code]){
            found_pattern[shifted_code] = true;
            this_pattern_is_new = true;
            m_lut(shifted_code) = lbp_code;
          }
        }
        if (this_pattern_is_new){
          ++lbp_code;
        }
      }
    }else{ // not rotation invariant
      // initialize LUT with non-special values
      if(m_add_average_bit && m_to_average)
        m_lut.resize(1 << (m_P+1));
      blitz::firstIndex i;
      m_lut = i;
    }
  }
}

int bob::ip::LBP::getMaxLabel() const {
  if (m_rotation_invariant){
    if (m_uniform)
      // rotation invariant uniform LBP
      return m_P + 2;
    else
      // rotation invariant non-uniform LBP
      // simply return the highest label plus 1
      return m_lut((1 << m_P) - 1) + 1;
  }else{
    if (m_uniform)
      // uniform LBP
      return m_P * (m_P-1) + 3;
    else{
      // regular LBP
      if (m_to_average && m_add_average_bit)
        return 1 << (m_P+1);
      else
        return 1 << m_P;
    }
  }
}
