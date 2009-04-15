#ifndef _TORCH5SPRO_WEAK_LEARNER_H_
#define _TORCH5SPRO_WEAK_LEARNER_H_

#include "Trainer.h"
#include "spCore.h"
#include "Machine.h"

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

	protected:

		////////////////////////////////////////////////////
		/// Attributes

		//
		long 	*m_shuffledindex_dataset;

		//
		double	*m_weights_dataset;

		// The machine that will be trained
		Machine 	*m_weak_classifier;

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
		spCore **m_features;
	};

}

#endif
