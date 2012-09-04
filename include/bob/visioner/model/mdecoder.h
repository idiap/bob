/**
 * @file visioner/visioner/model/mdecoder.h
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

#ifndef BOB_VISIONER_MDECODER_H
#define BOB_VISIONER_MDECODER_H

#include "visioner/model/loss.h"
#include "visioner/model/tagger.h"
#include "visioner/model/model.h"
#include "visioner/model/trainer.h"

namespace bob { namespace visioner {

  // Decode parameters
  boost::shared_ptr<Loss>		make_loss(const param_t& param);
  boost::shared_ptr<Tagger>	make_tagger(const param_t& param);
  boost::shared_ptr<Model>        make_model(const param_t& param);
  boost::shared_ptr<Trainer>	make_trainer(const param_t& param);

  OptimizationType        make_optimization(const param_t& param);
  FeatureSharingType      make_sharing(const param_t& param);

  // Retrieve the lists of encoded objects
  std::vector<std::string> available_losses_list();
  std::vector<std::string> available_taggers_list();
  std::vector<std::string> available_models_list();
  std::vector<std::string> available_trainers_list();
  std::vector<std::string> available_optimizations_list();
  std::vector<std::string> available_sharings_list();

  // Retrieve the lists of encoded objects as a single string
  std::string available_losses();
  std::string available_taggers();
  std::string available_models();
  std::string available_trainers();
  std::string available_optimizations();
  std::string available_sharings();

}}

#endif // BOB_VISIONER_MDECODER_H
