/**
 * @file bob/trainer/FisherLDATrainer.h
 * @date Sat Jun 4 21:38:59 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Implements a multi-class Fisher/LDA linear machine Training using
 * Singular Value Decomposition (SVD). For more information on Linear Machines
 * and associated methods, please consult Bishop, Machine Learning and Pattern
 * Recognition chapter 4.
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

#ifndef BOB_TRAINER_FISHER_LDA_TRAINER_H
#define BOB_TRAINER_FISHER_LDA_TRAINER_H

#include "Trainer.h"
#include <bob/machine/LinearMachine.h>
#include <vector>

namespace bob { namespace trainer { 
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief Sets a linear machine to perform the Fisher/LDA decomposition.\n
 *
 * References:\n
 * 1. Bishop, Machine Learning and Pattern Recognition chapter 4.\n
 * 2. http://en.wikipedia.org/wiki/Linear_discriminant_analysis
 */
class FisherLDATrainer: Trainer<bob::machine::LinearMachine, std::vector<blitz::Array<double,2> > >
{
  public: //api

    /**
     * @brief Initializes a new Fisher/LDA trainer. The training stage will
     * place the resulting fisher components in the linear machine and set it
     * up to extract the variable means automatically.
     */
    FisherLDATrainer();

    /**
     * @brief Destructor
     */
    virtual ~FisherLDATrainer();
    
    /**
     * @brief Copy constructor.
     */
    FisherLDATrainer(const FisherLDATrainer& other);

    /**
     * @brief Assignment operator
     */
    FisherLDATrainer& operator=(const FisherLDATrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const FisherLDATrainer& other) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const FisherLDATrainer& other) const;
   /**
     * @brief Similar to
     */
    bool is_similar_to(const FisherLDATrainer& other, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Trains the LinearMachine to perform Fisher/LDA discrimination.
     * The resulting machine will have the eigen-vectors of the 
     * Sigma-1 * Sigma_b product, arranged by decreasing energy.
     *
     * Each input arrayset represents data from a given input class.
     *
     * Note we set only the N-1 eigen vectors in the linear machine since the
     * last eigen value should be zero anyway. You can compress the machine
     * output further using resize() if necessary.
     */
    virtual void train(bob::machine::LinearMachine& machine, 
        const std::vector<blitz::Array<double,2> >& data);

    /**
     * @brief Trains the LinearMachine to perform Fisher/LDA discrimination.
     * The resulting machine will have the eigen-vectors of the 
     * Sigma-1 * Sigma_b product, arranged by decreasing energy. You don't
     * need to sort the results. Also returns the eigen values of the 
     * covariance matrix product so you can use that to choose which
     * components to keep.
     *
     * Each input arrayset represents data from a given input class.
     *
     * Note we set only the N-1 eigen vectors in the linear machine since the
     * last eigen value should be zero anyway. You can compress the machine
     * output further using resize() if necessary.
     */
    virtual void train(bob::machine::LinearMachine& machine,
        blitz::Array<double,1>& eigen_values,
        const std::vector<blitz::Array<double,2> >& data);

};

/**
 * @}
 */
}}

#endif /* BOB_TRAINER_FISHER_LDA_TRAINER_H */
