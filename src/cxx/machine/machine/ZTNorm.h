/**
 * @file cxx/machine/machine/ZTNorm.h
 * @date Tue Jul 19 15:33:20 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef TORCH_ZTNORM_H
#define TORCH_ZTNORM_H

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

#endif /* TORCH_ZTNORM_H */
