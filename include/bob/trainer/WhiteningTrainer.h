/**
 * @file bob/trainer/WhiteningTrainer.h
 * @date Tue Apr 2 21:04:00 2013 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_TRAINER_WHITENING_TRAINER_H
#define BOB_TRAINER_WHITENING_TRAINER_H

#include "Trainer.h"
#include <bob/machine/LinearMachine.h>
#include <blitz/array.h>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief Sets a linear machine to perform a Whitening transform\n
 *
 * Reference:\n
 * "Independent Component Analysis: Algorithms and Applications",
 *   Aapo Hyvärinen, Erkki Oja,
 *   Neural Networks, 2000, vol. 13, p. 411--430\n
 * 
 * Given a training set X, this will compute the W matrix such that:\n
 *   \f$W = cholesky(inv(cov(X_{n},X_{n}^{T})))\f$, where \f$X_{n}\f$
 *   corresponds to the center data
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
