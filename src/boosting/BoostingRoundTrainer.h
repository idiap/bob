#ifndef _TORCH5SPRO_BOOSTING_ROUND_TRAINER_H_
#define _TORCH5SPRO_BOOSTING_ROUND_TRAINER_H_

#include "DataSet.h"
#include "WeakLearner.h"
#include "measurer.h"
#include "BoostingTrainer.h"

namespace Torch
{

    //////////////////////////////////////////////////
    // Boosts already selected features
    /////////////////////////////////////////////////
    class BoostingRoundTrainer : public BoostingTrainer
    {
    public:
            BoostingRoundTrainer();
            virtual bool train();
            virtual ~BoostingRoundTrainer();
    private:
        int m_nrounds; //number of weakclassifiers are set in BoostingTrainer
        int *m_featuremask;   // list of features that has to used for selecting best weakfeature
        bool m_mask;            // says wether to use the mask or not.
        int m_features;
        //void cleanup();

};



}

#endif
