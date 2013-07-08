/**
 * @file bob/trainer/WCCNTrainer.h
 * @date Tue Apr 9 22:10:00 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
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

#ifndef BOB_TRAINER_WCCN_TRAINER_H
#define BOB_TRAINER_WCCN_TRAINER_H

#include "Trainer.h"
#include <bob/machine/LinearMachine.h>
#include <blitz/array.h>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief Sets a linear machine to perform a WCCN transform\n
 *
 * Reference 1:\n
 * "Within-class covariance normalization for SVM-based speaker recognition",
 *   Andrew O. Hatch, Sachin Kajarekar, and Andreas Stolcke,
 *   ICSLP, 2006\n
 * Reference 2:\n
 * "N. Dehah, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet",
 *   Front-end factor analysis for speaker verification,
 *   IEEE TASLP, 2011\n
 * 
 * Given a training set X, this will compute the W matrix such that:\n
 *   \f$W = cholesky(inv(cov(X_{n},X_{n}^{T})))\f$, where \f$X_{n}\f$
 *   corresponds to the center data
 */
class WCCNTrainer: public Trainer<bob::machine::LinearMachine, std::vector<blitz::Array<double,2> > >
{
  public:
    /**
     * @brief Initializes a new WCCN trainer.
     */
    WCCNTrainer();

    /**
     * @brief Copy constructor
     */
    WCCNTrainer(const WCCNTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~WCCNTrainer();

    /**
     * @brief Assignment operator
     */
    WCCNTrainer& operator=(const WCCNTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const WCCNTrainer& other) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const WCCNTrainer& other) const;
   /**
     * @brief Similar to
     */
    bool is_similar_to(const WCCNTrainer& other, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Trains the LinearMachine to perform the WCCN
     */
    virtual void train(bob::machine::LinearMachine& machine, 
        const std::vector<blitz::Array<double, 2> >& data);

  private: //representation
};

/**
 * @}
 */
}}

#endif /* BOB_TRAINER_WCCN_TRAINER_H */
