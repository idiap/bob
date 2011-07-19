#include <boost/python.hpp>
#include <machine/ZTNorm.h>

using namespace boost::python;

static boost::shared_ptr<blitz::Array<double, 2> > ztnorm1(blitz::Array<double, 2>& eval_tests_on_eval_models,
                                                           blitz::Array<double, 2>& znorm_tests_on_eval_models,
                                                           blitz::Array<double, 2>& eval_tests_on_tnorm_models,
                                                           blitz::Array<double, 2>& znorm_tests_on_tnorm_models,
                                                           blitz::Array<bool,   2>& znorm_tests_tnorm_models_same_spk_ids) {

  boost::shared_ptr<blitz::Array<double, 2> > ret(new blitz::Array<double, 2>);

  Torch::machine::ztNorm(eval_tests_on_eval_models,
                         znorm_tests_on_eval_models,
                         eval_tests_on_tnorm_models,
                         znorm_tests_on_tnorm_models,
                         znorm_tests_tnorm_models_same_spk_ids,
                         *ret.get());

  return ret;
}

static boost::shared_ptr<blitz::Array<double, 2> > ztnorm2(blitz::Array<double, 2>& eval_tests_on_eval_models,
                                                           blitz::Array<double, 2>& znorm_tests_on_eval_models,
                                                           blitz::Array<double, 2>& eval_tests_on_tnorm_models,
                                                           blitz::Array<double, 2>& znorm_tests_on_tnorm_models) {

  boost::shared_ptr<blitz::Array<double, 2> > ret(new blitz::Array<double, 2>);

  Torch::machine::ztNorm(eval_tests_on_eval_models,
                         znorm_tests_on_eval_models,
                         eval_tests_on_tnorm_models,
                         znorm_tests_on_tnorm_models,
                         *ret.get());

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