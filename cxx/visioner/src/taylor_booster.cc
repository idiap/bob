/**
 * @file visioner/src/taylor_booster.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include <boost/format.hpp>
#include <boost/thread.hpp>

#include "core/logging.h"

#include "visioner/util/timer.h"
#include "visioner/model/trainers/taylor_booster.h"
#include "visioner/model/trainers/lutproblems/lut_problem_ept.h"
#include "visioner/model/trainers/lutproblems/lut_problem_var.h"

namespace bob { namespace visioner {

  // Constructor
  TaylorBooster::TaylorBooster(const param_t& param)
    :	Trainer(param)
  {
  }

  // Train a model using the given training and validation samples
  bool TaylorBooster::train(
      const Sampler& t_sampler, const Sampler& v_sampler, Model& model,
      size_t threads)
  {
    const uint64_t n_boots = m_param.m_bootstraps;
    const uint64_t t_b_n_samples = m_param.m_train_samples * inverse(n_boots + 1);
    const uint64_t v_n_samples = m_param.m_valid_samples;
    const uint64_t b_n_rounds = std::max((uint64_t)1, m_param.m_rounds >> n_boots);

    std::vector<uint64_t> t_samples, t_b_samples, v_samples;
    DataSet t_data, v_data;

    Timer timer;

    // Bootstrap the training samples
    for (uint64_t b = 0; b <= n_boots; b ++)
    {
      // Sample training data (first uniformly and then bootstrapped)
      //      & validation data (always uniformly)
      timer.restart();

      //      --- sampling
      (b == 0) ?
        t_sampler.sample(t_b_n_samples, t_b_samples, threads) :
        t_sampler.sample(t_b_n_samples, model, t_b_samples, threads);
      t_samples.insert(t_samples.end(), t_b_samples.begin(), t_b_samples.end());

      v_sampler.sample(v_n_samples, v_samples, threads);
      
      TDEBUG1("Sampling time for step " << b << " (" << n_boots << " bootstraps) is " << timer.elapsed() << " seconds.");
      timer.restart();

      //      --- mapping
      t_sampler.map(t_samples, model, t_data, threads);
      v_sampler.map(v_samples, model, v_data, threads);

      TDEBUG1("Mapping time for step " << b << " (" << n_boots << " bootstraps) is " << timer.elapsed() << " seconds.");

      // Train the model
      timer.restart();

      Generalizer<std::vector<std::vector<LUT> > > gen;
      m_param.m_rounds = b_n_rounds << b;
      if (train(t_data, v_data, model, gen, threads) == false || model.set(gen.model()) == false) {
        throw std::runtime_error("TaylorBooster failed to train the model");
      }

      TDEBUG1("Training time for step " << b << " (" << n_boots << " bootstraps) is " << timer.elapsed() << " seconds.");
    }

    // OK
    return true;
  }

  // Train a model
  bool TaylorBooster::train(
      const DataSet& t_data, const DataSet& v_data, const Model& model,
      Generalizer<std::vector<std::vector<LUT> > >& gen, size_t threads) const
  {
    if (t_data.empty()) throw std::runtime_error("Empty training set");
    if (v_data.empty()) throw std::runtime_error("Empty validation set");
    if (t_data.n_outputs() < 1) {
      boost::format m("Number of outputs in training set is %d (< 1)");
      m % t_data.n_outputs();
      throw std::runtime_error(m.str());
    }
    if (t_data.n_features() < 1) {
      boost::format m("Number of features in training set is %d (< 1)");
      m % t_data.n_features();
      throw std::runtime_error(m.str());
    }
    if (t_data.n_outputs() != v_data.n_outputs()) {
      boost::format m("Number of outputs in training set (%d) is different than the one on the validation set (%d)");
      m % t_data.n_outputs() % v_data.n_outputs();
      throw std::runtime_error(m.str());
    }
    if (t_data.n_features() != v_data.n_features()) {
      boost::format m("Number of features in training set (%d) is different than the one on the validation set (%d)");
      m % t_data.n_features() % v_data.n_features();
      throw std::runtime_error(m.str());
    }

    TDEBUG1("Using " << t_data.n_samples() << " training and "
      << v_data.n_samples() << " validation samples with "
      << t_data.n_features() << " features to train "
      << m_param.m_rounds << " weak learners.");

    // Regularization factors
    static const double lambdas[] = { 0.0, 0.1, 0.2, 0.5, 1.0 };
    const uint64_t n_lambdas =
      (make_optimization(m_param) == Variational) ? sizeof(lambdas)/sizeof(double) : 1;

    // Tune the regularization factor ...
    for (uint64_t ilambda = 0; ilambda < n_lambdas; ilambda ++)
    {
      const double lambda = lambdas[ilambda];

      // Create the solvers
      boost::shared_ptr<LUTProblem> t_lp, v_lp;
      switch (make_optimization(m_param))
      {
        case Expectation:
          t_lp.reset(new LUTProblemEPT(t_data, m_param, threads));
          v_lp.reset(new LUTProblemEPT(v_data, m_param, threads));
          break;

        case Variational:
          t_lp.reset(new LUTProblemVAR(t_data, m_param, lambda, threads));
          v_lp.reset(new LUTProblemVAR(v_data, m_param, lambda, threads));
          break;
      }

      // And train the model
      const std::string base_description =
        "<<lambda " + boost::lexical_cast<std::string>(lambda) + ">>";

      train(t_lp, v_lp, base_description, model, gen);
    }

    // OK
    TDEBUG1("Optimal: " << gen.description() << ": train = " 
        << gen.train_error() << ", valid = " << gen.valid_error() << ".");

    return true;
  }

  // Train a model
  void TaylorBooster::train(
      const boost::shared_ptr<LUTProblem>& t_lp, const boost::shared_ptr<LUTProblem>& v_lp,
      const std::string& base_description, const Model& model, Generalizer<std::vector<std::vector<LUT> > >& gen) const
  {
    Timer timer;

    // Train the models in boosting rounds ...
    for (uint64_t nc = 0; nc < m_param.m_rounds; nc ++)
    {
      // Train weak learners ...
      timer.restart();
      t_lp->update_loss_deriv();
      t_lp->select();
#     ifdef BOB_DEBUG
      const double time_select = timer.elapsed();
#     endif

      timer.restart();
      if (t_lp->line_search() == false)
      {
        break;
      }
#     ifdef BOB_DEBUG
      const double time_optimize = timer.elapsed();
#     endif

      // Check the generalization properties of the model
      const std::string description = base_description +
        "<<round " + boost::lexical_cast<std::string>(nc + 1) + "/" +
        boost::lexical_cast<std::string>(m_param.m_rounds) + ">>";

      t_lp->update_scores(t_lp->luts());
      t_lp->update_loss();

      v_lp->update_scores(t_lp->luts());
      //v_lp->update_loss();

      gen.process(t_lp->error(), v_lp->error(), t_lp->mluts(), description);

      TDEBUG1(description
        << ": train = " << t_lp->value() << " / " << t_lp->error()
        << ", valid = " << v_lp->error()
        << " in " << time_select << "+" << time_optimize << "seconds.");

#     ifdef BOB_DEBUG
      for (uint64_t o = 0; o < t_lp->n_outputs(); o ++) {
        TDEBUG1("output <" << (o + 1) << "/" 
            << t_lp->n_outputs() << "> selected feature <" 
            << model.describe(t_lp->luts()[o].feature()) << ">.");
      }
#     endif

    }
  }

}}
