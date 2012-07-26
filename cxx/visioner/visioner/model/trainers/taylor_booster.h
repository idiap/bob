#ifndef BOB_VISIONER_TAYLOR_BOOSTER_H
#define BOB_VISIONER_TAYLOR_BOOSTER_H

#include "visioner/model/trainer.h"
#include "visioner/model/generalizer.h"
#include "visioner/model/mdecoder.h"
#include "visioner/model/trainers/lutproblems/lut_problem.h"

namespace bob { namespace visioner {        

  ////////////////////////////////////////////////////////////////////////////////
  // TaylorBooster: 
  //      greedy boosting of multivariate weak learners using 
  //      the local Taylor expansion of the loss
  //      in the functional space of the weak learners.
  ////////////////////////////////////////////////////////////////////////////////

  class TaylorBooster : public Trainer
  {
    public:

      // Constructor
      TaylorBooster(const param_t& param = param_t());

      // Destructor
      virtual ~TaylorBooster() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual rtrainer_t clone() const 
      {
        return rtrainer_t(new TaylorBooster(m_param)); 
      }

      // Train a model using the given training and validation samples
      virtual bool train(	
          const Sampler& t_sampler, const Sampler& v_sampler, Model& model);

    private:

      // Generalizer for LUTs
      typedef Generalizer<MultiLUTs>  GenModel;

      // Train a model
      bool train(const DataSet& t_data, const DataSet& v_data, 
          const Model& model, GenModel& gen) const;
      void train(const rlutproblem_t& t_lp, const rlutproblem_t& v_lp,
          const string_t& base_description, const Model& model, GenModel& gen) const;
  };

}}

#endif // BOB_VISIONER_TAYLOR_BOOSTER_H
