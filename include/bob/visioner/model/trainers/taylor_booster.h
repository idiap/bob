/**
 * @file visioner/visioner/model/trainers/taylor_booster.h
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

#ifndef BOB_VISIONER_TAYLOR_BOOSTER_H
#define BOB_VISIONER_TAYLOR_BOOSTER_H

#include "bob/visioner/model/trainer.h"
#include "bob/visioner/model/generalizer.h"
#include "bob/visioner/model/mdecoder.h"
#include "bob/visioner/model/trainers/lutproblems/lut_problem.h"

namespace bob { namespace visioner {        

  /**
   * TaylorBooster: greedy boosting of multivariate weak learners using the
   * local Taylor expansion of the loss in the functional space of the weak
   * learners.
   */
  class TaylorBooster : public Trainer {

    public:

      /**
       * Builds a new trainer using TaylorBoost that will train a new model
       * with the given number of threads.
       */
      TaylorBooster(const param_t& param = param_t());

      // Destructor
      virtual ~TaylorBooster() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) 
      { m_param = param; }

      // Clone the object
      virtual boost::shared_ptr<Trainer> clone() const {
        return boost::shared_ptr<Trainer>(new TaylorBooster(m_param)); 
      }

      /**
       * Trains a model using the given training and validation samples. The
       * number of threads control if the training will be executed on the
       * current thread (zero) or in separate threads (one or more), what can
       * considerably speed it up.
       */
      virtual bool train(const Sampler& t_sampler, const Sampler& v_sampler,
          Model& model, size_t threads=0);

    private:

      // Train a model
      bool train(const DataSet& t_data, const DataSet& v_data, 
          const Model& model, Generalizer<std::vector<std::vector<LUT> > >& gen,
          size_t threads) const;

      void train(const boost::shared_ptr<LUTProblem>& t_lp, const boost::shared_ptr<LUTProblem>& v_lp, const std::string& base_description, const Model& model, Generalizer<std::vector<std::vector<LUT> > >& gen) const;

  };

}}

#endif // BOB_VISIONER_TAYLOR_BOOSTER_H
