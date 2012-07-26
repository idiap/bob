#ifndef BOB_VISIONER_TAGGER_KEYPOINT_OXY_H
#define BOB_VISIONER_TAGGER_KEYPOINT_OXY_H

#include "visioner/model/tagger.h"

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Sub-window labelling for keypoint regression.
  //	Returns the Ox&Oy coordinates of all keypoints in <param.m_keypoints>.
  // NB. A valid sample overlaps the object with at least <param.m_min_gt_overlap> and it
  //	should contain all the target keypoints specified in <param.m_labels>.
  /////////////////////////////////////////////////////////////////////////////////////////

  class KeypointOxyTagger : public Tagger
  {
    public:

      // Constructor
      KeypointOxyTagger(const param_t& param = param_t());

      // Destructor
      virtual ~KeypointOxyTagger() {}

      // Clone the object
      virtual rtagger_t clone() const { return rtagger_t(new KeypointOxyTagger(*this)); }

      // Number of outputs
      virtual index_t n_outputs() const;

      // Number of types
      virtual index_t n_types() const { return 1; }

      // Label a sub-window
      virtual bool check(const ipscale_t& ipscale, int x, int y, 
          scalars_t& targets, index_t& type) const;
  };

}}

#endif // BOB_VISIONER_TAGGER_KEYPOINT_OXY_H
