#ifndef _TORCH5SPRO_STOCHASTIC_GRADIENT_TRAINER_H_
#define _TORCH5SPRO_STOCHASTIC_GRADIENT_TRAINER_H_

#include "core/Trainer.h"
#include "machine/Criterion.h"

namespace Torch
{
    class StochasticGradientTrainer : public Trainer
    {
    public:
        ///
        StochasticGradientTrainer();

        ///
        virtual bool train();

        ///
        virtual ~StochasticGradientTrainer();

	///
	bool 		setCriterion(Criterion *m_criterion_);
	
    protected:

        // shuffle index
        long *m_shuffledindex;

	Criterion *m_criterion;
    };
}

#endif
