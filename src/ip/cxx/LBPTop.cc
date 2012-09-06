/**
 * @file ip/cxx/LBPTop.cc
 * @date Tue Apr 26 19:20:57 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Tiago Freitas Pereira <Tiago.Pereira@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * This class can be used to calculate the LBP-Top  of a set of image frames
 * representing a video sequence (c.f. Dynamic Texture Recognition Using Local
 * Binary Patterns with an Application to Facial Expression from Zhao &
 * Pietikäinen, IEEE Trans. on PAMI, 2007). This is the implementation file.
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

#include "bob/ip/LBPTop.h"
#include "bob/ip/Exception.h"

namespace ip = bob::ip;

ip::LBPTop::LBPTop(const bob::ip::LBP& lbp_xy, 
                   const bob::ip::LBP& lbp_xt, 
                   const bob::ip::LBP& lbp_yt)
: m_lbp_xy(lbp_xy.clone()),
  m_lbp_xt(lbp_xt.clone()),
  m_lbp_yt(lbp_yt.clone())
{
 /*Checking the inputs. The radius in XY,XT and YT must be the same*/

  if(lbp_xy.getRadius()!=lbp_xt.getRadius())
    throw LBPRadiusDoesNotMatch("X","XY","XT");

  if(lbp_xy.getRadius2()!=lbp_yt.getRadius())
    throw LBPRadiusDoesNotMatch("Y","XY","YT");

  if(lbp_xt.getRadius2()!=lbp_yt.getRadius2())
    throw LBPRadiusDoesNotMatch("T","XT","YT");



}

ip::LBPTop::LBPTop(const LBPTop& other)
: m_lbp_xy(other.m_lbp_xy->clone()),
  m_lbp_xt(other.m_lbp_xt->clone()),
  m_lbp_yt(other.m_lbp_yt->clone())
{
}

ip::LBPTop::~LBPTop() { }

ip::LBPTop& ip::LBPTop::operator= (const LBPTop& other) {
  m_lbp_xy = other.m_lbp_xy->clone();
  m_lbp_xt = other.m_lbp_xt->clone();
  m_lbp_yt = other.m_lbp_yt->clone();
  return *this;
}

void ip::LBPTop::operator()(const blitz::Array<uint8_t,3>& src, 
    blitz::Array<uint16_t,3>& xy,
    blitz::Array<uint16_t,3>& xt,
    blitz::Array<uint16_t,3>& yt) const
{
  process<uint8_t>(src, xy, xt, yt);
}

void ip::LBPTop::operator()(const blitz::Array<uint16_t,3>& src, 
    blitz::Array<uint16_t,3>& xy,
    blitz::Array<uint16_t,3>& xt,
    blitz::Array<uint16_t,3>& yt) const
{ 
  process<uint16_t>(src, xy, xt, yt); 
}
void ip::LBPTop::operator()(const blitz::Array<double,3>& src, 
    blitz::Array<uint16_t,3>& xy,
    blitz::Array<uint16_t,3>& xt,
    blitz::Array<uint16_t,3>& yt) const
{ 
  process<double>(src, xy, xt, yt); 
}
