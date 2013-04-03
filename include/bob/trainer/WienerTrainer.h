/**
 * @file bob/trainer/WienerTrainer.h
 * @date Fri Sep 30 16:58:42 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Trainer for the Wiener Machine
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

#ifndef BOB_TRAINER_WIENER_TRAINER_H
#define BOB_TRAINER_WIENER_TRAINER_H

#include "Trainer.h"
#include <bob/machine/WienerMachine.h>
#include <blitz/array.h>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief Sets a Wiener machine to perform a Wiener filtering, using the
 * Fourier statistics of a given dataset.\n
 *
 * Reference:\n
 * "Computer Vision: Algorithms and Applications", Richard Szeliski
 * (Part 3.4.3)
 */
class WienerTrainer: Trainer<bob::machine::WienerMachine, blitz::Array<double,3> >
{
  public: //api
    /**
     * @brief Default constructor.
     * Initializes a new Wiener trainer. 
     */
    WienerTrainer();

    /**
     * @brief Copy constructor
     */
    WienerTrainer(const WienerTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~WienerTrainer();

    /**
     * @brief Assignment operator
     */
    WienerTrainer& operator=(const WienerTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const WienerTrainer& other) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const WienerTrainer& other) const;
   /**
     * @brief Similar to
     */
    bool is_similar_to(const WienerTrainer& other, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Trains the WienerMachine to perform the filtering. 
     */
    virtual void train(bob::machine::WienerMachine& machine, 
        const blitz::Array<double,3>& data);

  private: //representation
};

/**
 * @}
 */
}}

#endif /* BOB_TRAINER_WIENER_TRAINER_H */
