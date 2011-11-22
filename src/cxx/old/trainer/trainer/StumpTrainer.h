/**
 * @file cxx/old/trainer/trainer/StumpTrainer.h
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
#ifndef _TORCH5SPRO_STUMP_TRAINER_H_
#define _TORCH5SPRO_STUMP_TRAINER_H_

#include "trainer/WeakLearner.h"
#include "machine/StumpMachine.h"
#include "measurer/measurer.h"

namespace Torch
{

    class StumpTrainer : public WeakLearner
    {
    public:
    	///
        StumpTrainer(StumpMachine *stump_machine_, int n_features_, spCore **features_ = NULL);

	///
        virtual bool train();

	///
        virtual ~StumpTrainer();

    private:

	//
	float *features_values;
	int n_bins;
	float **histogram;
	float **cumulative_histogram;

	bool verbose;
	StumpMachine *m_stump_machine;
    };

}

#endif
