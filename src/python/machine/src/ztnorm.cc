/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Sun 20 Nov 20:27:26 2011 CET
 *
 * @brief Binds ZT-normalization to python
 */

#include "core/python/ndarray.h"

#include <boost/python.hpp>
#include <machine/ZTNorm.h>

using namespace boost::python;
namespace tp = Torch::python;

static object ztnorm1(
    tp::const_ndarray eval_tests_on_eval_models,
    tp::const_ndarray znorm_tests_on_eval_models,
    tp::const_ndarray eval_tests_on_tnorm_models,
    tp::const_ndarray znorm_tests_on_tnorm_models,
    tp::const_ndarray znorm_tests_tnorm_models_same_spk_ids) {

  blitz::Array<double, 2> ret;

  blitz::Array<double,2> eval_tests_on_eval_models_ = 
    eval_tests_on_eval_models.bz<double,2>();
  blitz::Array<double,2> znorm_tests_on_eval_models_ =
    znorm_tests_on_eval_models.bz<double,2>();
  blitz::Array<double,2> eval_tests_on_tnorm_models_ = 
    eval_tests_on_tnorm_models.bz<double,2>();
  blitz::Array<double,2> znorm_tests_on_tnorm_models_ =
    znorm_tests_on_tnorm_models.bz<double,2>();
  blitz::Array<bool,2> znorm_tests_tnorm_models_same_spk_ids_ =
    znorm_tests_tnorm_models_same_spk_ids.bz<bool,2>();

  Torch::machine::ztNorm(eval_tests_on_eval_models_,
                         znorm_tests_on_eval_models_,
                         eval_tests_on_tnorm_models_,
                         znorm_tests_on_tnorm_models_,
                         znorm_tests_tnorm_models_same_spk_ids_,
                         ret);

  return object(ret); //full copy!
}

static object ztnorm2(
    tp::const_ndarray eval_tests_on_eval_models,
    tp::const_ndarray znorm_tests_on_eval_models,
    tp::const_ndarray eval_tests_on_tnorm_models,
    tp::const_ndarray znorm_tests_on_tnorm_models) {

  blitz::Array<double,2> eval_tests_on_eval_models_ = 
    eval_tests_on_eval_models.bz<double,2>();
  blitz::Array<double,2> znorm_tests_on_eval_models_ =
    znorm_tests_on_eval_models.bz<double,2>();
  blitz::Array<double,2> eval_tests_on_tnorm_models_ = 
    eval_tests_on_tnorm_models.bz<double,2>();
  blitz::Array<double,2> znorm_tests_on_tnorm_models_ =
    znorm_tests_on_tnorm_models.bz<double,2>();

  blitz::Array<double, 2> ret; //full copy!

  Torch::machine::ztNorm(eval_tests_on_eval_models_,
                         znorm_tests_on_eval_models_,
                         eval_tests_on_tnorm_models_,
                         znorm_tests_on_tnorm_models_,
                         ret);

  return object(ret);
}

void bind_machine_ztnorm() {
  def("ztnorm",
      ztnorm1,
      args("eval_tests_on_eval_models",
           "znorm_tests_on_eval_models",
           "eval_tests_on_tnorm_models",
           "znorm_tests_on_tnorm_models",
           "znorm_tests_tnorm_models_same_spk_ids"),
      "Normalize the evaluation scores with ZT-Norm"
     );
  
  def("ztnorm",
      ztnorm2,
      args("eval_tests_on_eval_models",
           "znorm_tests_on_eval_models",
           "eval_tests_on_tnorm_models",
           "znorm_tests_on_tnorm_models"),
      "Normalize the evaluation scores with ZT-Norm. Assume that znorm and tnorm have no common subject id."
     );
}
