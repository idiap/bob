#ifndef _TORCH5SPRO_LBP_ROUND_TRAINER_H_
#define _TORCH5SPRO_LBP_ROUND_TRAINER_H_

#include "WeakLearner.h"
#include "IntLutMachine.h"

namespace Torch
{

    class LBPRoundTrainer : public WeakLearner
    {
    public:
        ///
        LBPRoundTrainer(IntLutMachine *lbp_machine_, int n_features_, spCore **features_ = NULL);
     //   virtual double forward(Tensor *example_);
        ///
        virtual bool train();

        ///
        virtual ~LBPRoundTrainer();
        int getLUTSize();
        double *getLUT();

    private:

        int *features_values;
        int n_bins;
        float **histogram;
        //double *lut_;
        //
        IntLutMachine *m_lbp_machine;
        int b_lutsize;
        double *bestlut_;
    };

}

#endif
