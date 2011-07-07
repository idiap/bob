/**
 * @author Andre Anjos <andre.anjos@idiap.ch> 
 * @date Wed 06 Jul 2011 16:38:39 CEST 
 *
 * @brief Stopping criteria for MLP training
 */

#ifndef TORCH_TRAINER_MLPSTOPCRITERIA_H 
#define TORCH_TRAINER_MLPSTOPCRITERIA_H

#include "machine/MLP.h"

namespace Torch { namespace trainer {

  /**
   * The stop criteria defines when to stop the neural network training. It
   * receives the trained MLP and the current training iteration number and
   * decides to stop the training or not. The stop criteria can be any callable
   * method that respects the API:
   *
   * bool function(const Torch::machine::MLP& network, size_t iteration) { }
   *
   * This means it can be a stateful object such as an instantiated class or
   * just some iteration number check, or even a lambda function.
   */
  typedef boost::function<bool (const Torch::machine::MLP& network,
      size_t iteration)> MLPStopCriteria;

  /**
   * A criteria that just regulates the number of iterations for training an
   * MLP.
   */
  class NumberOfIterationsCriteria {

    public: //api

      /**
       * Give the maximum number of iterations to train for.
       */
      NumberOfIterationsCriteria(size_t n);

      /**
       * D'tor virtualisation
       */
      virtual ~NumberOfIterationsCriteria();

      /**
       * Set and get
       */
      size_t getN() const { return m_n; }
      void setN(size_t n) const { m_n = n; }

      /**
       * Checks the number of iterations and report a should stop flag back.
       */
      bool operator()(const Torch::machine::MLP& network, 
          size_t iteration) const;

    private: //representation

      size_t m_n; ///< number of iterations value

  };
  

#endif /* TORCH_TRAINER_MLPSTOPCRITERIA_H */

