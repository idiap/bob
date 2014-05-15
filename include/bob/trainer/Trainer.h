/**
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */
#ifndef BOB_TRAINER_TRAINER_H
#define BOB_TRAINER_TRAINER_H

/**
 * @addtogroup TRAINER trainer
 * @brief Trainer module API
 */
namespace bob {
/**
 * @ingroup TRAINER
 */
namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief Root class for all trainers
 */
template<class T_machine, class T_sampler>
class Trainer
{
public:
  virtual ~Trainer() {};

  /**
   * @brief Train a \c machine using a sampler
   *
   * @param machine machine to train
   * @param sampler sampler that provides training data
   */
  virtual void train(T_machine& machine, const T_sampler& sampler) = 0;
};

/**
 * @}
 */
}}

#endif // BOB_TRAINER_TRAINER_H
