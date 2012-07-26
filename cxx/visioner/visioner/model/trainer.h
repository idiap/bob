#ifndef BOB_VISIONER_TRAINER_H
#define BOB_VISIONER_TRAINER_H

#include "sampler.h"
#include "model.h"

namespace bob { namespace visioner {

  class Trainer;
  typedef boost::shared_ptr<Trainer>	rtrainer_t;

  ////////////////////////////////////////////////////////////////////////////////
  // Trains a model using the given training and validation samples.
  ////////////////////////////////////////////////////////////////////////////////
  class Trainer : public Parametrizable
  {
    public:	

      // Constructor
      Trainer(const param_t& param = param_t())
        :	Parametrizable(param)
      {			
      }

      // Clone the object
      virtual rtrainer_t clone() const = 0;

      // Train a model using the given training and validation samples
      virtual bool train(	
          const Sampler& t_sampler, const Sampler& v_sampler, Model& model) = 0;
  };

}}

#endif // BOB_VISIONER_TRAINER_H
