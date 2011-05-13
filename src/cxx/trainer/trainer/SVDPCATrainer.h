#ifndef TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H
#define TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H

#include "Trainer.h"
#include "core/logging.h"
#include "machine/EigenMachine.h"
#include "math/svd.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace Torch 
{
  namespace trainer 
  {
  
    class SVDPCATrainer : virtual public Trainer<Torch::machine::EigenMachine, Torch::machine::FrameSample>
    {
      public:
        SVDPCATrainer() {}
        virtual ~SVDPCATrainer() {}
  
        void train(Torch::machine::EigenMachine& machine, const Sampler<Torch::machine::FrameSample>& data); 

      protected:

    };

  }
}

#endif /* TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H */
