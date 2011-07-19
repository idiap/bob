#ifndef _ZTNORM_H
#define _ZTNORM_H

#include <blitz/array.h>
#include <machine/GMMMachine.h>
#include <vector>

namespace Torch { namespace machine {

  /**
   * Normalize the evaluation scores with ZT-Norm
   *
   * @exception Torch::core::UnexpectedShapeError matrix sizes are not consistent
   * 
   * @param eval_tests_on_eval_models
   * @param znorm_tests_on_eval_models
   * @param eval_tests_on_tnorm_models
   * @param znorm_tests_on_tnorm_models
   * @param znorm_tests_tnorm_models_same_spk_ids
   * @param[out] scores normalized scores
   */
  void ztNorm(blitz::Array<double, 2>& eval_tests_on_eval_models,
              blitz::Array<double, 2>& znorm_tests_on_eval_models,
              blitz::Array<double, 2>& eval_tests_on_tnorm_models,
              blitz::Array<double, 2>& znorm_tests_on_tnorm_models,
              blitz::Array<bool,   2>& znorm_tests_tnorm_models_same_spk_ids,
              blitz::Array<double, 2>& scores);
  
  /**
   * Normalize the evaluation scores with ZT-Norm.
   * Assume that znorm and tnorm have no common subject id.
   *
   * @exception Torch::core::UnexpectedShapeError matrix sizes are not consistent
   *
   * @param eval_tests_on_eval_models
   * @param znorm_tests_on_eval_models
   * @param eval_tests_on_tnorm_models
   * @param znorm_tests_on_tnorm_models
   * @param[out] scores normalized scores
   */
  void ztNorm(blitz::Array<double, 2>& eval_tests_on_eval_models,
              blitz::Array<double, 2>& znorm_tests_on_eval_models,
              blitz::Array<double, 2>& eval_tests_on_tnorm_models,
              blitz::Array<double, 2>& znorm_tests_on_tnorm_models,
              blitz::Array<double, 2>& scores);
}
}

#endif // _ZTNORM_H