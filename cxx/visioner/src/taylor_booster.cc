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
      const Sampler& t_sampler, const Sampler& v_sampler, Model& model)
  {
    const index_t n_boots = m_param.m_bootstraps;
    const index_t t_b_n_samples = m_param.m_train_samples * inverse(n_boots + 1);
    const index_t v_n_samples = m_param.m_valid_samples;  
    const index_t b_n_rounds = std::max((index_t)1, m_param.m_rounds >> n_boots);

    indices_t t_samples, t_b_samples, v_samples;
    DataSet t_data, v_data;

    Timer timer;

    // Bootstrap the training samples
    for (index_t b = 0; b <= n_boots; b ++)
    {                
      // Sample training data (first uniformly and then bootstrapped)
      //      & validation data (always uniformly)
      timer.restart();

      //      --- sampling
      (b == 0) ? 
        t_sampler.sample(t_b_n_samples, t_b_samples) :
        t_sampler.sample(t_b_n_samples, model, t_b_samples);
      t_samples.insert(t_samples.end(), t_b_samples.begin(), t_b_samples.end());

      v_sampler.sample(v_n_samples, v_samples);

      //      --- mapping
      t_sampler.map(t_samples, model, t_data);
      v_sampler.map(v_samples, model, v_data);

      bob::core::info << "timing: sampling ~ " << timer.elapsed() << "." << std::endl;

      // Train the model
      timer.restart();

      GenModel gen;
      m_param.m_rounds = b_n_rounds << b;
      if (    train(t_data, v_data, model, gen) == false ||
          model.set(gen.model()) == false)
      {
        bob::core::error << "Failed to train the model!" << std::endl;
        return false;
      }

      bob::core::info << "timing: training ~ " << timer.elapsed() << "." << std::endl;
    }

    // OK
    return true;
  }

  // Train a model
  bool TaylorBooster::train(
      const DataSet& t_data, const DataSet& v_data, 
      const Model& model, GenModel& gen) const
  {
    // Check parameters
    if (	t_data.empty() || v_data.empty() ||
        t_data.n_outputs() < 1 ||
        t_data.n_outputs() != v_data.n_outputs() ||
        t_data.n_features() != v_data.n_features() ||
        t_data.n_features() < 1)
    {
      bob::core::error << "Invalid training & validation samples!" << std::endl;
      return false;
    }

    bob::core::info
      << "using "
      << t_data.n_samples() << " training and "
      << v_data.n_samples() << " validation samples with "
      << t_data.n_features() << " features to train "
      << m_param.m_rounds << " weak learners." << std::endl; 

    // Regularization factors
    static const scalar_t lambdas[] = { 0.0, 0.1, 0.2, 0.5, 1.0 };
    const index_t n_lambdas = 
      (make_optimization(m_param) == Variational) ? sizeof(lambdas)/sizeof(scalar_t) : 1;

    // Tune the regularization factor ...
    for (index_t ilambda = 0; ilambda < n_lambdas; ilambda ++)
    {
      const scalar_t lambda = lambdas[ilambda];

      // Create the solvers 
      rlutproblem_t t_lp, v_lp;
      switch (make_optimization(m_param))
      {
        case Expectation:
          t_lp.reset(new LUTProblemEPT(t_data, m_param));
          v_lp.reset(new LUTProblemEPT(v_data, m_param));
          break;

        case Variational:
          t_lp.reset(new LUTProblemVAR(t_data, m_param, lambda));
          v_lp.reset(new LUTProblemVAR(v_data, m_param, lambda));
          break;
      }

      // And train the model                        
      const string_t base_description =
        "<<lambda " + boost::lexical_cast<string_t>(lambda) + ">>";

      train(t_lp, v_lp, base_description, model, gen);
    }

    // OK
    bob::core::info
      << "optimal: " << gen.description()
      << ": train = " << gen.train_error()
      << ", valid = " << gen.valid_error() << "." << std::endl;

    return true;
  }

  // Train a model
  void TaylorBooster::train(
      const rlutproblem_t& t_lp, const rlutproblem_t& v_lp, 
      const string_t& base_description, const Model& model, GenModel& gen) const
  {
    Timer timer;

    // Train the models in boosting rounds ...
    for (index_t nc = 0; nc < m_param.m_rounds; nc ++)
    {                        
      // Train weak learners ...
      timer.restart();                        
      t_lp->update_loss_deriv();
      t_lp->select();                        
      const scalar_t time_select = timer.elapsed();

      timer.restart();                        
      if (t_lp->line_search() == false)
      {
        break;
      }                            
      const scalar_t time_optimize = timer.elapsed();

      // Check the generalization properties of the model
      const string_t description = base_description +
        "<<round " + boost::lexical_cast<string_t>(nc + 1) + "/" +
        boost::lexical_cast<string_t>(m_param.m_rounds) + ">>";

      t_lp->update_scores(t_lp->luts());
      t_lp->update_loss();                        

      v_lp->update_scores(t_lp->luts());
      //v_lp->update_loss();

      gen.process(t_lp->error(), v_lp->error(), t_lp->mluts(), description);

      // Debug
      bob::core::info
        << description
        << ": train = " << t_lp->value() << " / " << t_lp->error()
        << ", valid = " << v_lp->error()
        << " in " << time_select << "+" << time_optimize << "s." << std::endl;                        

      // Debug
      for (index_t o = 0; o < t_lp->n_outputs(); o ++)
      {
        const LUT& lut = t_lp->luts()[o];
        bob::core::info
          << "output <" << (o + 1) << "/" << t_lp->n_outputs() 
          << "> selected feature <" << model.describe(lut.feature()) << ">." << std::endl;
      }
    }
  }

}}
