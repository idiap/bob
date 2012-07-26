#ifndef BOB_VISIONER_MDECODER_H
#define BOB_VISIONER_MDECODER_H

#include "visioner/model/loss.h"
#include "visioner/model/tagger.h"
#include "visioner/model/model.h"
#include "visioner/model/trainer.h"

namespace bob { namespace visioner {

  // Decode parameters
  rloss_t		make_loss(const param_t& param);
  rtagger_t	make_tagger(const param_t& param);
  rmodel_t        make_model(const param_t& param);
  rtrainer_t	make_trainer(const param_t& param);

  OptimizationType        make_optimization(const param_t& param);
  FeatureSharingType      make_sharing(const param_t& param);

  // Retrieve the lists of encoded objects
  string_t        available_losses();
  string_t        available_taggers();
  string_t        available_models();
  string_t        available_trainers();

  string_t        available_optimizations();
  string_t        available_sharings();

}}

#endif // BOB_VISIONER_MDECODER_H
