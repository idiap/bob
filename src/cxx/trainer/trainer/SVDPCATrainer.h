#ifndef TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H
#define TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H

#include "Trainer.h"
#include "core/logging.h"
#include "database/Arrayset.h"
#include "machine/EigenMachine.h"
#include "math/svd.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace Torch 
{
  namespace trainer 
  {
  
    class SVDPCATrainer : virtual public Trainer<Torch::machine::EigenMachine, Torch::database::Arrayset>
    {
      public:
        SVDPCATrainer() {}
        virtual ~SVDPCATrainer() {}
  
        void train(Torch::machine::EigenMachine& machine, const Torch::database::Arrayset& data); 

      protected:

    };

  }
}

#endif /* TORCH5SPRO_TRAINER_SVDPCA_TRAINER_H */
