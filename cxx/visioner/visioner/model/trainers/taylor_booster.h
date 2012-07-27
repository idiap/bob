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

#include "visioner/model/trainer.h"
#include "visioner/model/generalizer.h"
#include "visioner/model/mdecoder.h"
#include "visioner/model/trainers/lutproblems/lut_problem.h"

namespace bob { namespace visioner {        

  ////////////////////////////////////////////////////////////////////////////////
  // TaylorBooster: 
  //      greedy boosting of multivariate weak learners using 
  //      the local Taylor expansion of the loss
  //      in the functional space of the weak learners.
  ////////////////////////////////////////////////////////////////////////////////

  class TaylorBooster : public Trainer
  {
    public:

      // Constructor
      TaylorBooster(const param_t& param = param_t());

      // Destructor
      virtual ~TaylorBooster() {}

      // Reset to new parameters
      virtual void reset(const param_t& param) { m_param = param; }

      // Clone the object
      virtual rtrainer_t clone() const 
      {
        return rtrainer_t(new TaylorBooster(m_param)); 
      }

      // Train a model using the given training and validation samples
      virtual bool train(	
          const Sampler& t_sampler, const Sampler& v_sampler, Model& model);

    private:

      // Generalizer for LUTs
      typedef Generalizer<MultiLUTs>  GenModel;

      // Train a model
      bool train(const DataSet& t_data, const DataSet& v_data, 
          const Model& model, GenModel& gen) const;
      void train(const rlutproblem_t& t_lp, const rlutproblem_t& v_lp,
          const string_t& base_description, const Model& model, GenModel& gen) const;
  };

}}

#endif // BOB_VISIONER_TAYLOR_BOOSTER_H
