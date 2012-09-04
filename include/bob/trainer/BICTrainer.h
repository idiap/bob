/**
 * @file cxx/trainer/trainer/BICTrainer.h
 * @date Wed Jun  6 10:29:09 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
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

#ifndef BOB_TRAINER_BICTRAINER_H
#define BOB_TRAINER_BICTRAINER_H

#include <machine/BICMachine.h>
#include <io/Arrayset.h>

namespace bob { namespace trainer {

  class BICTrainer {
    public:
      //! initializes a BICTrainer to train IEC (without subspace estimation)
      BICTrainer() : m_M_I(0), m_M_E(0) {}
      //! initializes a BICTrainer to train BIC (including subspace truncation)
      BICTrainer(int intra_dim, int extra_dim) : m_M_I(intra_dim), m_M_E(extra_dim) {}

      //! trains the intrapersonal and extrapersonal classes of the given BICMachine
      void train(bob::machine::BICMachine& machine, const bob::io::Arrayset& intra_differences, const bob::io::Arrayset& extra_differences) const {
        train_single(false, machine, intra_differences);
        train_single(true, machine, extra_differences);
      }

      //! trains the intrapersonal or the extrapersonal class of the given BICMachine
      void train_single(bool clazz, bob::machine::BICMachine& machine, const bob::io::Arrayset& differences) const;

    private:

      //! dimensions of the intrapersonal and extrapersonal subspace;
      //! zero if training IEC.
      int m_M_I, m_M_E;
  };

}}


#endif // BOB_TRAINER_BICTRAINER_H
