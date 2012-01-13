/**
 * @file src/python/machine/src/ztnorm.cc
 * @date Tue Jul 19 15:33:20 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds ZT-normalization to python
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

#include "core/python/ndarray.h"

#include <boost/python.hpp>
#include "machine/ZTNorm.h"

using namespace boost::python;
namespace tp = bob::python;
namespace ca = bob::core::array;

static object ztnorm1(
    tp::const_ndarray eval_tests_on_eval_models,
    tp::const_ndarray znorm_tests_on_eval_models,
    tp::const_ndarray eval_tests_on_tnorm_models,
    tp::const_ndarray znorm_tests_on_tnorm_models,
    tp::const_ndarray znorm_tests_tnorm_models_same_spk_ids) 
{
  const blitz::Array<double,2> eval_tests_on_eval_models_ = 
    eval_tests_on_eval_models.bz<double,2>();
  const blitz::Array<double,2> znorm_tests_on_eval_models_ =
    znorm_tests_on_eval_models.bz<double,2>();
  const blitz::Array<double,2> eval_tests_on_tnorm_models_ = 
    eval_tests_on_tnorm_models.bz<double,2>();
  const blitz::Array<double,2> znorm_tests_on_tnorm_models_ =
    znorm_tests_on_tnorm_models.bz<double,2>();
  const blitz::Array<bool,2> znorm_tests_tnorm_models_same_spk_ids_ =
    znorm_tests_tnorm_models_same_spk_ids.bz<bool,2>();

  // allocate output
  tp::ndarray ret(ca::t_float64, eval_tests_on_eval_models_.extent(0), eval_tests_on_eval_models_.extent(1));
  blitz::Array<double, 2> ret_ = ret.bz<double,2>();

  bob::machine::ztNorm(eval_tests_on_eval_models_,
                         znorm_tests_on_eval_models_,
                         eval_tests_on_tnorm_models_,
                         znorm_tests_on_tnorm_models_,
                         znorm_tests_tnorm_models_same_spk_ids_,
                         ret_);

  return ret.self();
}

static object ztnorm2(
    tp::const_ndarray eval_tests_on_eval_models,
    tp::const_ndarray znorm_tests_on_eval_models,
    tp::const_ndarray eval_tests_on_tnorm_models,
    tp::const_ndarray znorm_tests_on_tnorm_models) 
{
  const blitz::Array<double,2> eval_tests_on_eval_models_ = 
    eval_tests_on_eval_models.bz<double,2>();
  const blitz::Array<double,2> znorm_tests_on_eval_models_ =
    znorm_tests_on_eval_models.bz<double,2>();
  const blitz::Array<double,2> eval_tests_on_tnorm_models_ = 
    eval_tests_on_tnorm_models.bz<double,2>();
  const blitz::Array<double,2> znorm_tests_on_tnorm_models_ =
    znorm_tests_on_tnorm_models.bz<double,2>();

  // allocate output
  tp::ndarray ret(ca::t_float64, eval_tests_on_eval_models_.extent(0), eval_tests_on_eval_models_.extent(1));
  blitz::Array<double, 2> ret_ = ret.bz<double,2>();

  bob::machine::ztNorm(eval_tests_on_eval_models_,
                         znorm_tests_on_eval_models_,
                         eval_tests_on_tnorm_models_,
                         znorm_tests_on_tnorm_models_,
                         ret_);

  return ret.self();
}

void bind_machine_ztnorm() 
{
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
