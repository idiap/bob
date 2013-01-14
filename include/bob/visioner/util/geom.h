/**
 * @file bob/visioner/util/geom.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#ifndef BOB_VISIONER_GEOM_H
#define BOB_VISIONER_GEOM_H

#include <QRectF>
#include <QPointF>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdint.h>

namespace bob { namespace visioner {

  // Sub-window (location + scale)
  struct subwindow_t {
    subwindow_t(int x = 0, int y = 0, int s = 0)
      :	m_x(x), m_y(y), m_s(s)
    {
    }
    uint16_t       m_x, m_y;
    uint32_t       m_s;
  };

  // Region of interest
  struct roi_t {
    roi_t(int min_x, int max_x, int min_y, int max_y)
      :	m_min_x(min_x), m_max_x(max_x), m_min_y(min_y), m_max_y(max_y)
    {
    }

    int16_t		m_min_x, m_max_x;
    int16_t		m_min_y, m_max_y;
  };

  // Area of a rectangle	
  inline qreal area(const QRectF& rect) {
    return rect.width() * rect.height();
  }	

  // Compare two sub-windows
  inline bool operator<(const subwindow_t& one, const subwindow_t& two) {
    return one.m_s < two.m_s ||
      (one.m_s == two.m_s && one.m_x < two.m_x) ||
      (one.m_x == two.m_x && one.m_y < two.m_y);
  }

}}

// Compare two rectangles
inline bool operator<(const QRectF& one, const QRectF& two) {
  return one.top() < two.top() || (one.top() == two.top() && 
    (one.left() < two.left()  || (one.left() == two.left() && 
    (one.width() < two.width() || (one.width() == two.width() && 
    one.height() < two.height())))));
}

#endif // BOB_VISIONER_GEOM_H
