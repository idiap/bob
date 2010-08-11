#ifndef _TORCH5SPRO_LBP_ROUND_TRAINER_H_
#define _TORCH5SPRO_LBP_ROUND_TRAINER_H_

#include "trainer/WeakLearner.h"
#include "machine/IntLutMachine.h"

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
        int getLUTSize() { return m_b_lutsize;}

        double *getLUT() { return m_bestlut_; }

    private:

        int *m_features_values;
        int m_n_bins;
        float **m_histogram;
        //double *lut_;
        //
        IntLutMachine *m_lbp_machine;
        int m_b_lutsize;
        double *m_bestlut_;
        bool verbose;
    };

}

#endif
