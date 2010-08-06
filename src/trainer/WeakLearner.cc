#include "WeakLearner.h"

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
