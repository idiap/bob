#ifndef TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H
#define TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H

#include "trainer/Trainer.h"
#include "machine/EigenMachine.h"
#include "machine/FrameClassificationSample.h"

namespace Torch 
{
  namespace trainer 
  {
  
    class FisherLDATrainer : virtual public Trainer<Torch::machine::EigenMachine, Torch::machine::FrameClassificationSample>
    {
      public:
        /**
          * @brief Constructor
          * The number of different classes, labeled from 0 to n_classes-1
          */
        FisherLDATrainer(int n_classes): m_n_classes(n_classes) {}
        /**
          * @brief Destructor
          */
        virtual ~FisherLDATrainer() {}
  
        /**
          * @brief Start the training process
          */
        void train(Torch::machine::EigenMachine& machine, const Sampler<Torch::machine::FrameClassificationSample>& data); 

      private:
        void computeScatterMatrices( const Sampler<Torch::machine::FrameClassificationSample>& data);

        int m_n_classes;
        blitz::Array<double,2> m_Sb; // Between-class scatter matrix
        blitz::Array<double,2> m_Sw; // Within-class scatter matrix

    };

  }
}

#endif /* TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H */
