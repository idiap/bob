/**
 * @file visioner/cxx/tagger_object.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#include "bob/visioner/model/taggers/tagger_object.h"

namespace bob { namespace visioner {

  // Constructor
  ObjectTagger::ObjectTagger(Type type, const param_t& param)
    :	Tagger(param), m_type(type)
  {
  }

  // Number of outputs
  uint64_t ObjectTagger::n_outputs() const
  {
    return m_param.m_labels.size();
  }

  // Label a sub-window
  bool ObjectTagger::check(const ipscale_t& ipscale, int x, int y, 
      std::vector<double>& targets, uint64_t& type) const
  {
    // Background image: all SWs are valid negative samples
    if (ipscale.m_objects.empty() == true)
    {
      type = 0;
      std::fill(targets.begin(), targets.end(), neg_target());
      return true;
    }

    // Image with objects: check the overlapping with the ground truth
    else
    {
      const QRectF reg(x, y, m_param.m_cols, m_param.m_rows);

      int oindex = 0;
      const double overlap = visioner::overlap(reg, ipscale.m_objects, &oindex);

      // Valid positive sample
      if (overlap >= m_param.m_min_gt_overlap)
      {
        // Of the correct object type/view/id?!
        const int lindex = find(ipscale.m_objects[oindex]);
        if (lindex >= 0)
        {                                
          type = lindex + 1;
          std::fill(targets.begin(), targets.end(), neg_target());
          targets[lindex] = pos_target();
          return true;
        }
        else
        {
          return false;
        }
      }

      //                        // Valid negative sample
      //                        else if (overlap <= 1.0 - m_param.m_min_gt_overlap)
      //                        {
      //                                type = 0;
      //                                return true;
      //                        }

      // Invalid sample
      else
      {
        return false;
      }
    }
  }

}}
