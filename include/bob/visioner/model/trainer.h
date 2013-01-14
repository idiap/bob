/**
 * @file bob/visioner/model/trainer.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_VISIONER_TRAINER_H
#define BOB_VISIONER_TRAINER_H

#include "sampler.h"
#include "model.h"

namespace bob { namespace visioner {

  /**
   * Trains a model using the given training and validation samples.
   */
  class Trainer : public Parametrizable {

    public:

      // Constructor
      Trainer(const param_t& param = param_t())
        :	Parametrizable(param) {	}

      // Clone the object
      virtual boost::shared_ptr<Trainer> clone() const = 0;

      /**
       * Train a model using the given training and validation samples. The
       * number of threads control if the training will be executed on the
       * current thread (zero) or in separate threads (one or more), what can
       * considerably speed it up.
       */
      virtual bool train(const Sampler& t_sampler, const Sampler& v_sampler, 
          Model& model, size_t threads) = 0;
  };

}}

#endif // BOB_VISIONER_TRAINER_H
