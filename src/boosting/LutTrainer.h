#ifndef _TORCH5SPRO_LUT_TRAINER_H_
#define _TORCH5SPRO_LUT_TRAINER_H_

#include "WeakLearner.h"
#include "LutMachine.h"

namespace Torch
{

    class LutTrainer : public WeakLearner
    {
    public:
    	///
        LutTrainer(LutMachine *lut_machine_, int n_features_, spCore **features_ = NULL);

	///
        virtual bool train();

	///
        virtual ~LutTrainer();
	
    private:

	//
	LutMachine *m_lut_machine;
    };

}

#endif
