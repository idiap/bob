/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 29 Sep 2011
 *
 * @brief Trainer for the Wiener Machine
 */

#ifndef TORCH5SPRO_TRAINER_WIENER_TRAINER_H
#define TORCH5SPRO_TRAINER_WIENER_TRAINER_H

#include "machine/WienerMachine.h"
#include "io/Arrayset.h"

namespace Torch { namespace trainer {
  
  /**
   * Sets a Wiener machine to perform a Wiener filtering, using the Fourier
   * statistics of a given dataset.
   *
   * Computer Vision: Algorithms and Applications, Richard Szeliski
   * (Part 3.4.3)
   */
  class WienerTrainer {

    public: //api

      /**
       * Default constructor
       * Initializes a new Wiener trainer. 
       */
      WienerTrainer();

      /**
       * Copy construction.
       */
      WienerTrainer(const WienerTrainer& other);

      /**
       * Destructor virtualisation
       */
      virtual ~WienerTrainer();

      /**
       * Copy operator
       */
      WienerTrainer& operator=(const WienerTrainer& other);

      /**
       * Trains the WienerMachine to perform the filtering. 
       */
      virtual void train(Torch::machine::WienerMachine& machine, 
          const Torch::io::Arrayset& data) const;

    private: //representation

  };

}}

#endif /* TORCH5SPRO_TRAINER_WIENER_TRAINER_H */
