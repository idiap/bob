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

#include <stdexcept>
#include <boost/format.hpp>
#include <bob/ip/LBPTop.h>

bob::ip::LBPTop::LBPTop(const bob::ip::LBP& lbp_xy,
                   const bob::ip::LBP& lbp_xt,
                   const bob::ip::LBP& lbp_yt)
: m_lbp_xy(lbp_xy),
  m_lbp_xt(lbp_xt),
  m_lbp_yt(lbp_yt)
{
  /*
   * Checking the inputs. The radius in XY,XT and YT must be the same
   */
  if(lbp_xy.getRadii()[0]!=lbp_xt.getRadii()[0]) {
    boost::format m("the radii R_xy[0] (%f) and R_xt[0] (%f) do not match");
    m % lbp_xy.getRadii()[0] % lbp_xt.getRadii()[0];
    throw std::runtime_error(m.str());
  }

  if(lbp_xy.getRadii()[1]!=lbp_yt.getRadii()[0]) {
    boost::format m("the radii R_xy[1] (%f) and R_yt[0] (%f) do not match");
    m % lbp_xy.getRadii()[1] % lbp_yt.getRadii()[0];
    throw std::runtime_error(m.str());
  }

  if(lbp_xt.getRadii()[1]!=lbp_yt.getRadii()[0]) {
    boost::format m("the radii R_xt[1] (%f) and R_yt[0] (%f) do not match");
    m % lbp_xt.getRadii()[1] % lbp_yt.getRadii()[0];
    throw std::runtime_error(m.str());
  }

}

bob::ip::LBPTop::LBPTop(const LBPTop& other)
: m_lbp_xy(other.m_lbp_xy),
  m_lbp_xt(other.m_lbp_xt),
  m_lbp_yt(other.m_lbp_yt)
{
}

bob::ip::LBPTop::~LBPTop() { }

bob::ip::LBPTop& bob::ip::LBPTop::operator= (const LBPTop& other) {
  m_lbp_xy = other.m_lbp_xy;
  m_lbp_xt = other.m_lbp_xt;
  m_lbp_yt = other.m_lbp_yt;
  return *this;
}

void bob::ip::LBPTop::operator()(const blitz::Array<uint8_t,3>& src,
    blitz::Array<uint16_t,3>& xy,
    blitz::Array<uint16_t,3>& xt,
    blitz::Array<uint16_t,3>& yt) const
{
  process<uint8_t>(src, xy, xt, yt);
}

void bob::ip::LBPTop::operator()(const blitz::Array<uint16_t,3>& src,
    blitz::Array<uint16_t,3>& xy,
    blitz::Array<uint16_t,3>& xt,
    blitz::Array<uint16_t,3>& yt) const
{
  process<uint16_t>(src, xy, xt, yt);
}
void bob::ip::LBPTop::operator()(const blitz::Array<double,3>& src,
    blitz::Array<uint16_t,3>& xy,
    blitz::Array<uint16_t,3>& xt,
    blitz::Array<uint16_t,3>& yt) const
{
  process<double>(src, xy, xt, yt);
}
