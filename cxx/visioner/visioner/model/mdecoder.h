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
  rloss_t		make_loss(const param_t& param);
  rtagger_t	make_tagger(const param_t& param);
  rmodel_t        make_model(const param_t& param);
  rtrainer_t	make_trainer(const param_t& param);

  OptimizationType        make_optimization(const param_t& param);
  FeatureSharingType      make_sharing(const param_t& param);

  // Retrieve the lists of encoded objects
  strings_t available_losses_list();
  strings_t available_taggers_list();
  strings_t available_models_list();
  strings_t available_trainers_list();
  strings_t available_optimizations_list();
  strings_t available_sharings_list();

  // Retrieve the lists of encoded objects as a single string
  string_t available_losses();
  string_t available_taggers();
  string_t available_models();
  string_t available_trainers();
  string_t available_optimizations();
  string_t available_sharings();

}}

#endif // BOB_VISIONER_MDECODER_H
