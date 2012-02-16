/**
 * @file cxx/trainer/trainer/Trainer.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
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
#ifndef TRAINER_H
#define TRAINER_H

namespace bob {
namespace trainer {

/**
 * Root class for all trainers
 */
template<class T_machine, class T_sampler>
class Trainer
{
public:
  virtual ~Trainer() {};

  /**
   * Train a \c machine using a sampler
   *
   * @param machine machine to train
   * @param sampler sampler that provides training data
   */
  virtual void train (T_machine& machine, const T_sampler& sampler) = 0;
};

}
}
#endif // TRAINER_H
