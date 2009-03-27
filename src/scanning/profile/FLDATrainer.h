#ifndef _TORCHVISION_SCANNING_PROFILE_FLDA_TRAINER_H_
#define _TORCHVISION_SCANNING_PROFILE_FLDA_TRAINER_H_

#include "Trainer.h"		// <FLDATrainer> is a <Trainer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::FLDATrainer:
	//	- trains FLDAMachine only on Nx1D DoubleTensors
        //
        //      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class FLDATrainer : public Torch::Trainer
	{
	public:

		// Constructor
		FLDATrainer();

		// Destructor
		virtual ~FLDATrainer();

		// Train the given machine on the given dataset
		virtual bool 		train();

        private:

                /////////////////////////////////////////////////////////////////
      		// Attributes

      		//
	};
}

#endif
