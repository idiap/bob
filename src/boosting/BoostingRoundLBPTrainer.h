#ifndef _TORCH5SPRO_BOOSTING_ROUND_LBP_TRAINER_H_
#define _TORCH5SPRO_BOOSTING_ROUND_LBP_TRAINER_H_

#include "DataSet.h"
#include "WeakLearner.h"
#include "measurer.h"
#include "BoostingTrainer.h"
#include "LBPRoundTrainer.h"
#include "IntLutMachine.h"
#include "Machine.h"

namespace Torch
{

    //////////////////////////////////////////////////
    // Boosts already selected features
    /////////////////////////////////////////////////
    class BoostingRoundLBPTrainer : public BoostingTrainer
    {
    public:
        BoostingRoundLBPTrainer();
        virtual bool train();
        virtual double forward(Tensor *example_);
        virtual ~BoostingRoundLBPTrainer();
        virtual bool setWeakLearners(int n_classifiers_, WeakLearner **weak_learners_);
        //    virtual void updateWeights();
    private:
        int m_nrounds;          //number of weakclassifiers are set in BoostingTrainer
        int *m_featuremask;     // list of features that has to used for selecting best weakfeature
        bool m_mask;            // says wether to use the mask or not.
        int m_features;
        int *m_trackfeatures;
        //void cleanup();
        void compressmachines();
        double ** m_lut_t1;
        bool verbose;

    };



}

#endif
