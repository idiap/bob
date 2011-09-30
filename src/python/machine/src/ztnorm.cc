#include <boost/python.hpp>
#include <machine/ZTNorm.h>

#include "core/python/pycore.h"

using namespace boost::python;
namespace tp = Torch::python;

static blitz::Array<double, 2> ztnorm1(
    numeric::array eval_tests_on_eval_models,
    numeric::array znorm_tests_on_eval_models,
    numeric::array eval_tests_on_tnorm_models,
    numeric::array znorm_tests_on_tnorm_models,
    numeric::array znorm_tests_tnorm_models_same_spk_ids) {

  blitz::Array<double,2> eval_tests_on_eval_models_ = 
    tp::numpy_bz<double,2>(eval_tests_on_eval_models);
  blitz::Array<double,2> znorm_tests_on_eval_models_ =
    tp::numpy_bz<double,2>(znorm_tests_on_eval_models);
  blitz::Array<double,2> eval_tests_on_tnorm_models_ = 
    tp::numpy_bz<double,2>(eval_tests_on_tnorm_models);
  blitz::Array<double,2> znorm_tests_on_tnorm_models_ =
    tp::numpy_bz<double,2>(znorm_tests_on_tnorm_models);
  blitz::Array<bool,2> znorm_tests_tnorm_models_same_spk_ids_ = 
    tp::numpy_bz<bool,2>(znorm_tests_tnorm_models_same_spk_ids);

  blitz::Array<double, 2> ret;

  Torch::machine::ztNorm(eval_tests_on_eval_models_,
                         znorm_tests_on_eval_models_,
                         eval_tests_on_tnorm_models_,
                         znorm_tests_on_tnorm_models_,
                         znorm_tests_tnorm_models_same_spk_ids_,
                         ret);

  return ret;
}

static blitz::Array<double, 2> ztnorm2(
    numeric::array eval_tests_on_eval_models,
    numeric::array znorm_tests_on_eval_models,
    numeric::array eval_tests_on_tnorm_models,
    numeric::array znorm_tests_on_tnorm_models) {

  blitz::Array<double,2> eval_tests_on_eval_models_ = 
    tp::numpy_bz<double,2>(eval_tests_on_eval_models);
  blitz::Array<double,2> znorm_tests_on_eval_models_ =
    tp::numpy_bz<double,2>(znorm_tests_on_eval_models);
  blitz::Array<double,2> eval_tests_on_tnorm_models_ = 
    tp::numpy_bz<double,2>(eval_tests_on_tnorm_models);
  blitz::Array<double,2> znorm_tests_on_tnorm_models_ =
    tp::numpy_bz<double,2>(znorm_tests_on_tnorm_models);

  blitz::Array<double, 2> ret;

  Torch::machine::ztNorm(eval_tests_on_eval_models_,
                         znorm_tests_on_eval_models_,
                         eval_tests_on_tnorm_models_,
                         znorm_tests_on_tnorm_models_,
                         ret);

  return ret;
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
