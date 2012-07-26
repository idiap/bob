#include "visioner/model/taggers/tagger_keypoint_oxy.h"

namespace bob { namespace visioner {

  // Constructor
  KeypointOxyTagger::KeypointOxyTagger(const param_t& param)
    :	Tagger(param)
  {
  }

  // Number of outputs
  index_t KeypointOxyTagger::n_outputs() const
  {
    return 2 * m_param.m_labels.size();
  }

  // Label a sub-window
  bool KeypointOxyTagger::check(const ipscale_t& ipscale, int x, int y, 
      scalars_t& targets, index_t& type) const
  {
    const rect_t reg(x, y, m_param.m_cols, m_param.m_rows);
    const scalar_t inv_x = inverse(m_param.m_cols);
    const scalar_t inv_y = inverse(m_param.m_rows);

    // Valid if it overlaps a large part of the object ...
    for (objects_t::const_iterator it = ipscale.m_objects.begin();
        it != ipscale.m_objects.end(); ++ it)
    {
      const scalar_t overlap = visioner::overlap(reg, it->bbx());
      if (overlap >= m_param.m_min_gt_overlap)
      {
        Keypoint keypoint;
        bool valid = true;
        for (index_t i = 0; i < m_param.m_labels.size() && valid == true; i ++)
        {
          valid = it->find(m_param.m_labels[i], keypoint);
        }

        if (valid == true)
        {
          // OK, return the normalized Ox/Oy coordinates
          for (index_t i = 0; i < m_param.m_labels.size() && valid == true; i ++)
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
