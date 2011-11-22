/**
 * @file cxx/old/trainer/src/WeakLearner.cc
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
#include "trainer/WeakLearner.h"

namespace Torch
{
	WeakLearner::WeakLearner(Machine *weak_classifier_, int n_features_, spCore **features_)
	{
        	m_shuffledindex_dataset = NULL;
        	m_weights_dataset = NULL;

		m_machine = weak_classifier_;
		m_weak_classifier = weak_classifier_;

		m_n_features = n_features_;
		m_features = features_;
		m_mask =false;
	}

	bool WeakLearner::setDataShuffledIndex(long *shuffledindex_dataset_)
	{
	   	if(shuffledindex_dataset_ == NULL) return false;

        	m_shuffledindex_dataset = shuffledindex_dataset_;

        	return true;
    	}

	bool WeakLearner::setDataWeights(double *weights_dataset_)
	{
	   	if(weights_dataset_ == NULL) return false;

        	m_weights_dataset = weights_dataset_;

        	return true;
 	}

	WeakLearner::~WeakLearner()
	{

	}

}
