#ifndef TRAINER_H
#define TRAINER_H

namespace Torch {
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
