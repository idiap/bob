#ifndef _TORCHVISION_SCANNING_PROFILE_LR_TRAINER_H_
#define _TORCHVISION_SCANNING_PROFILE_LR_TRAINER_H_

#include "Trainer.h"		// <LRTrainer> is a <Trainer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::LRTrainer:
	//	- trains LRMachines only on Nx1D DoubleTensors
	//
        //      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class LRTrainer : public Torch::Trainer
	{
	public:

		// Constructor
		LRTrainer();

		// Destructor
		virtual ~LRTrainer();

		/// Train the given machine on the given dataset
		virtual bool 		train();

        private:

		// Compute the loglikelihood for the given dataset
		double			getLLH(const double* weights, int size) const;

                /////////////////////////////////////////////////////////////////
      		// Attributes

		//
	};
}

#endif
