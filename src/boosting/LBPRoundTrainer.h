#ifndef _TORCH5SPRO_LBP_ROUND_TRAINER_H_
#define _TORCH5SPRO_LBP_ROUND_TRAINER_H_

#include "WeakLearner.h"
#include "LBPMachine.h"

namespace Torch
{

    class LBPRoundTrainer : public WeakLearner
    {
    public:
    	///
        LBPRoundTrainer(LBPMachine *lbp_machine_, int n_features_, spCore **features_ = NULL);

	///
        virtual bool train();

	///
        virtual ~LBPRoundTrainer();

    private:

    int *features_values;
	int n_bins;
	float **histogram;
	//double *lut_;
	//
	LBPMachine *m_lbp_machine;
    };

}

#endif
