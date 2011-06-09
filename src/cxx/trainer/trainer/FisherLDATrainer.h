#ifndef TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H
#define TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H

#include "trainer/Trainer.h"
#include "machine/EigenMachine.h"
#include "database/Arrayset.h"
#include <vector>

namespace Torch 
{
  namespace trainer 
  {
  
    class FisherLDATrainer : virtual public Trainer<Torch::machine::EigenMachine, std::vector<Torch::database::Arrayset> >
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
        void train(Torch::machine::EigenMachine& machine, const std::vector<Torch::database::Arrayset>& sampler); 

      private:
        void computeScatterMatrices( const std::vector<Torch::database::Arrayset>& sampler);

        int m_n_classes;
        blitz::Array<double,2> m_Sb; // Between-class scatter matrix
        blitz::Array<double,2> m_Sw; // Within-class scatter matrix

    };

  }
}

#endif /* TORCH5SPRO_TRAINER_FISHER_LDA_TRAINER_H */
