#include "visioner/model/mdecoder.h"
#include "visioner/util/manager.h"

#include "visioner/model/losses/diag_exp_loss.h"
#include "visioner/model/losses/diag_log_loss.h"
#include "visioner/model/losses/diag_symexp_loss.h"
#include "visioner/model/losses/diag_symlog_loss.h"
#include "visioner/model/losses/jesorsky_loss.h"

#include "visioner/model/trainers/taylor_booster.h"
#include "visioner/model/trainers/averager.h"

#include "visioner/model/taggers/tagger_object.h"
#include "visioner/model/taggers/tagger_keypoint_oxy.h"

#include "visioner/model/models/mblbp_model.h"
#include "visioner/model/models/model_pool.h"

namespace bob { namespace visioner {

  typedef Manager<Loss>			LossManager;
  typedef Manager<Tagger>			TaggerManager;
  typedef Manager<Model>                  ModelManager;
  typedef Manager<Trainer>		TrainerManager;

  typedef Manager<Manageable<OptimizationType> >          OptimizationManager;
  typedef Manager<Manageable<FeatureSharingType> >        FeatureSharingManager;

  // Register the available object types
  static bool register_objects()
  {
    // Register boosted loss types
    OptimizationManager::get_mutable_instance().add("ept", Expectation);
    OptimizationManager::get_mutable_instance().add("var", Variational);

    // Register feature sharing types
    FeatureSharingManager::get_mutable_instance().add("shared", Shared);
    FeatureSharingManager::get_mutable_instance().add("indep", Independent);

    // Register the losses
    LossManager::get_mutable_instance().add("diag_exp", DiagExpLoss());
    LossManager::get_mutable_instance().add("diag_log", DiagLogLoss());
    LossManager::get_mutable_instance().add("diag_symlog", DiagSymLogLoss());
    LossManager::get_mutable_instance().add("diag_symexp", DiagSymExpLoss());
    LossManager::get_mutable_instance().add("jesorsky", JesorskyLoss());

    // Register the SW taggers
    TaggerManager::get_mutable_instance().add("object_type", ObjectTagger(ObjectTagger::TypeTagger));
    TaggerManager::get_mutable_instance().add("object_pose", ObjectTagger(ObjectTagger::PoseTagger));
    TaggerManager::get_mutable_instance().add("object_id", ObjectTagger(ObjectTagger::IDTagger));
    TaggerManager::get_mutable_instance().add("keypoint", KeypointOxyTagger());

    // Register models
    ModelManager::get_mutable_instance().add("lbp", 
        MBLBPModel());

    ModelManager::get_mutable_instance().add("elbp",
        ModelPool<MBLBPModel, 
        ModelPool<MBmLBPModel, 
        ModelPool<MBtLBPModel, MBdLBPModel> > >());

    ModelManager::get_mutable_instance().add("mct", 
        MBMCTModel());

    // Register trainers
    TrainerManager::get_mutable_instance().add("avg", Averager());                
    TrainerManager::get_mutable_instance().add("gboost", TaylorBooster());

    return true;
  }

  static const bool registered = register_objects();

  // Decode parameters
  rloss_t	make_loss(const param_t& param)
  {
    const rloss_t loss = LossManager::get_const_instance().get(param.m_loss);
    loss->reset(param);
    return loss;
  }

  rtagger_t make_tagger(const param_t& param)
  {
    rtagger_t tagger = TaggerManager::get_const_instance().get(param.m_tagger);
    tagger->reset(param);
    return tagger;
  }

  rmodel_t make_model(const param_t& param)
  {
    rmodel_t model = ModelManager::get_const_instance().get(param.m_feature);
    model->reset(param);
    return model;
  }

  rtrainer_t make_trainer(const param_t& param)
  {
    rtrainer_t trainer = TrainerManager::get_const_instance().get(param.m_trainer);
    trainer->reset(param);
    return trainer;
  }

  OptimizationType make_optimization(const param_t& param)
  {
    return **OptimizationManager::get_const_instance().get(param.m_optimization);
  }

  FeatureSharingType make_sharing(const param_t& param)
  {
    return **FeatureSharingManager::get_const_instance().get(param.m_sharing);
  }

  // Retrieve the lists of encoded objects
  string_t available_losses()
  {
    return LossManager::get_const_instance().describe();
  }

  string_t available_taggers()
  {
    return TaggerManager::get_const_instance().describe();
  }

  string_t available_models()
  {
    return ModelManager::get_const_instance().describe();
  }

  string_t available_trainers()
  {
    return TrainerManager::get_const_instance().describe();
  }

  string_t available_optimizations()
  {
    return OptimizationManager::get_const_instance().describe();
  }

  string_t available_sharings()
  {
    return FeatureSharingManager::get_const_instance().describe();
  }

}}
