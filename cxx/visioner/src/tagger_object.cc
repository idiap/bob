#include "visioner/model/taggers/tagger_object.h"

namespace bob { namespace visioner {

  // Constructor
  ObjectTagger::ObjectTagger(Type type, const param_t& param)
    :	Tagger(param), m_type(type)
  {
  }

  // Number of outputs
  index_t ObjectTagger::n_outputs() const
  {
    return m_param.m_labels.size();
  }

  // Label a sub-window
  bool ObjectTagger::check(const ipscale_t& ipscale, int x, int y, 
      scalars_t& targets, index_t& type) const
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
      const rect_t reg(x, y, m_param.m_cols, m_param.m_rows);

      int oindex = 0;
      const scalar_t overlap = visioner::overlap(reg, ipscale.m_objects, &oindex);

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
