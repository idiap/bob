#ifndef _TORCH5SPRO_GRADIENT_TRAINER_H_
#define _TORCH5SPRO_GRADIENT_TRAINER_H_

#include "core/Trainer.h"
#include "machine/Criterion.h"

namespace Torch
{
    class GradientTrainer : public Trainer
    {
    public:
        ///
        GradientTrainer();

        ///
        virtual bool train();

        ///
        virtual ~GradientTrainer();

	///
	bool 		setCriterion(Criterion *m_criterion_);
	
    protected:

	Criterion *m_criterion;
    };
}

#endif
