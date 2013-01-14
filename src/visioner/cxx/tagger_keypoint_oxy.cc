/**
 * @file visioner/cxx/tagger_keypoint_oxy.cc
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

#include "bob/visioner/model/taggers/tagger_keypoint_oxy.h"

namespace bob { namespace visioner {

  // Constructor
  KeypointOxyTagger::KeypointOxyTagger(const param_t& param)
    :	Tagger(param)
  {
  }

  // Number of outputs
  uint64_t KeypointOxyTagger::n_outputs() const
  {
    return 2 * m_param.m_labels.size();
  }

  // Label a sub-window
  bool KeypointOxyTagger::check(const ipscale_t& ipscale, int x, int y, 
      std::vector<double>& targets, uint64_t& type) const
  {
    const QRectF reg(x, y, m_param.m_cols, m_param.m_rows);
    const double inv_x = inverse(m_param.m_cols);
    const double inv_y = inverse(m_param.m_rows);

    // Valid if it overlaps a large part of the object ...
    for (std::vector<Object>::const_iterator it = ipscale.m_objects.begin();
        it != ipscale.m_objects.end(); ++ it)
    {
      const double overlap = visioner::overlap(reg, it->bbx());
      if (overlap >= m_param.m_min_gt_overlap)
      {
        Keypoint keypoint;
        bool valid = true;
        for (uint64_t i = 0; i < m_param.m_labels.size() && valid == true; i ++)
        {
          valid = it->find(m_param.m_labels[i], keypoint);
        }

        if (valid == true)
        {
          // OK, return the normalized Ox/Oy coordinates
          for (uint64_t i = 0; i < m_param.m_labels.size() && valid == true; i ++)
          {
            it->find(m_param.m_labels[i], keypoint);
            targets[2 * i + 0] = inv_x * (keypoint.m_point.x() - x);
            targets[2 * i + 1] = inv_y * (keypoint.m_point.y() - y);
          }

          type = 0;

          return true;
        }
      }
    }

    // Invalid sub-window
    return false;
  }

}}
