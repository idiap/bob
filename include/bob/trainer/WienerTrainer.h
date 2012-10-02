/**
 * @file bob/trainer/WienerTrainer.h
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Trainer for the Wiener Machine
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

#ifndef BOB5SPRO_TRAINER_WIENER_TRAINER_H
#define BOB5SPRO_TRAINER_WIENER_TRAINER_H

#include "bob/machine/WienerMachine.h"

namespace bob { namespace trainer {
  
  /**
   * Sets a Wiener machine to perform a Wiener filtering, using the Fourier
   * statistics of a given dataset.
   *
   * Computer Vision: Algorithms and Applications, Richard Szeliski
   * (Part 3.4.3)
   */
  class WienerTrainer {

    public: //api

      /**
       * Default constructor
       * Initializes a new Wiener trainer. 
       */
      WienerTrainer();

      /**
       * Copy construction.
       */
      WienerTrainer(const WienerTrainer& other);

      /**
       * Destructor virtualisation
       */
      virtual ~WienerTrainer();

      /**
       * Copy operator
       */
      WienerTrainer& operator=(const WienerTrainer& other);

      /**
       * Trains the WienerMachine to perform the filtering. 
       */
      virtual void train(bob::machine::WienerMachine& machine, 
          const blitz::Array<double,3>& data) const;

    private: //representation

  };

}}

#endif /* BOB5SPRO_TRAINER_WIENER_TRAINER_H */
