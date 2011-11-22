/**
 * @file cxx/old/trainer/trainer/BoostingRoundTrainer.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCH5SPRO_BOOSTING_ROUND_TRAINER_H_
#define _TORCH5SPRO_BOOSTING_ROUND_TRAINER_H_

#include "core/DataSet.h"
#include "trainer/WeakLearner.h"
#include "measurer/measurer.h"
#include "trainer/BoostingTrainer.h"

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
