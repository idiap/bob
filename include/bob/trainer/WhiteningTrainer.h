/**
 * @file bob/trainer/WhiteningTrainer.h
 * @date Tue Apr 2 21:04:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#ifndef BOB_TRAINER_WHITENING_TRAINER_H
#define BOB_TRAINER_WHITENING_TRAINER_H

#include "Trainer.h"
#include <bob/machine/LinearMachine.h>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief Sets a linear machine to perform a Whitening transform 
 *
 * Reference:
 * TODO
 */
class WhiteningTrainer: public Trainer<bob::machine::LinearMachine, blitz::Array<double,2> >
{
  public:
    /**
     * @brief Initializes a new Whitening trainer.
     */
    WhiteningTrainer();

    /**
     * @brief Copy constructor
     */
    WhiteningTrainer(const WhiteningTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~WhiteningTrainer();

    /**
     * @brief Assignment operator
     */
    WhiteningTrainer& operator=(const WhiteningTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const WhiteningTrainer& other) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const WhiteningTrainer& other) const;
   /**
     * @brief Similar to
     */
    bool is_similar_to(const WhiteningTrainer& other, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Trains the LinearMachine to perform the Whitening
     */
    virtual void train(bob::machine::LinearMachine& machine, 
        const blitz::Array<double,2>& data);

  private: //representation
};

/**
 * @}
 */
}}

#endif /* BOB_TRAINER_WHITENING_TRAINER_H */
