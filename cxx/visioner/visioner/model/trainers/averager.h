#ifndef BOB_VISIONER_AVERAGER_H
#define ABOB_VISIONER_VERAGER_H

#include "visioner/model/trainer.h"

namespace bob { namespace visioner {

  ////////////////////////////////////////////////////////////////////////////////
  // Construct LUT models with discrete, positive and bounded feature values
  //	with fix outputs: the average target values (for each output).
  ////////////////////////////////////////////////////////////////////////////////

  class Averager : public Trainer
  {
    public:

      // Constructor
      Averager(const param_t& param = param_t());

      // Destructor
      virtual ~Averager() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual rtrainer_t clone() const { return rtrainer_t(new Averager(m_param)); }

      // Train a model using the given training and validation samples
      virtual bool train(	
          const Sampler& t_sampler, const Sampler& v_sampler, Model& model);
  };

}}

#endif // BOB_VISIONER_AVERAGER_H
