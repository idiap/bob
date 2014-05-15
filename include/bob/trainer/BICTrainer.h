/**
 * @date Wed Jun  6 10:29:09 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_TRAINER_BICTRAINER_H
#define BOB_TRAINER_BICTRAINER_H

#include <bob/machine/BICMachine.h>

namespace bob { namespace trainer {
  /**
   * @ingroup TRAINER
   * @{
   */

  class BICTrainer {
    public:
      //! initializes a BICTrainer to train IEC (without subspace estimation)
      BICTrainer() : m_M_I(0), m_M_E(0) {}
      //! initializes a BICTrainer to train BIC (including subspace truncation)
      BICTrainer(int intra_dim, int extra_dim) : m_M_I(intra_dim), m_M_E(extra_dim) {}

      //! trains the intrapersonal and extrapersonal classes of the given BICMachine
      void train(bob::machine::BICMachine& machine, const blitz::Array<double,2>& intra_differences, const blitz::Array<double,2>& extra_differences) const {
        train_single(false, machine, intra_differences);
        train_single(true, machine, extra_differences);
      }

      //! trains the intrapersonal or the extrapersonal class of the given BICMachine
      void train_single(bool clazz, bob::machine::BICMachine& machine, const blitz::Array<double,2>& differences) const;

    private:

      //! dimensions of the intrapersonal and extrapersonal subspace;
      //! zero if training IEC.
      int m_M_I, m_M_E;
  };

  /**
   * @}
   */
}}


#endif // BOB_TRAINER_BICTRAINER_H
