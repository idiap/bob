/**
 * @file cxx/old/trainer/trainer/WeakLearner.h
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
#ifndef _TORCH5SPRO_WEAK_LEARNER_H_
#define _TORCH5SPRO_WEAK_LEARNER_H_

#include "trainer/FTrainer.h"
#include "sp/spCore.h"
#include "machine/Machine.h"

namespace Torch {

	class Machine;

	class WeakLearner : public Trainer
	{
	public:
		/// Constructor
		WeakLearner(Machine *weak_classifier_, int n_features_, spCore **features_ = NULL);

		///
		bool setDataShuffledIndex(long *shuffledindex_dataset_);

		///
		bool setDataWeights(double *weights_dataset_);

		///
		bool setWeight(double weight_) { m_weight = weight_; return true;}

		///
		double getWeight() { return m_weight; }


        //added by venkatesh..................................................
		bool setMask(bool m_mask_){m_mask = m_mask_; return true;}

		bool setFeatureMask(int *m_featuremask_) { m_featuremask = m_featuremask_; return true;}
		int getFeatureID() {return m_featureID;}
		int getmnFeatures(){return m_n_features;}
        //..........................................................................
		/// Destructor
		virtual ~WeakLearner();
		Machine 	*m_weak_classifier;
		// The bank of all possible features
		spCore **m_features;

	protected:

		////////////////////////////////////////////////////
		/// Attributes

		//
		long 	*m_shuffledindex_dataset;

		//
		double	*m_weights_dataset;

		// The machine that will be trained


		// The weight associated to the weak classifier
		double m_weight;

        //added by venkatesh
		int *m_featuremask; //required for boosting with rounds
		bool m_mask;
		int m_featureID;
        //.....................................................

		///
        	int m_n_features;

		// The bank of all possible features
	//	spCore **m_features;
	};

}

#endif
