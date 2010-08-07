#ifndef _TORCH5SPRO_EM_TRAINER_H_
#define _TORCH5SPRO_EM_TRAINER_H_

#include "core/Trainer.h"

namespace Torch
{
    class EMTrainer : public Trainer
    {
    public:
        ///
        EMTrainer();

        ///
        virtual bool train();

        ///
        virtual ~EMTrainer();
    };
}

#endif
