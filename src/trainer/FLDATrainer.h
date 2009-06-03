#ifndef _TORCHVISION_FLDA_TRAINER_H_
#define _TORCHVISION_FLDA_TRAINER_H_

#include "Trainer.h"		// <FLDATrainer> is a <Trainer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::FLDATrainer:
	//	- trains FLDAMachine only on Nx1D DoubleTensors
	//	- the DataSet's targets are considered positive if their values are >= 0.0
	//
	//	- if a validation dataset is provided then it will be used to set
	//		the optimum threshold, otherwise the default threshold (0.0) will be used
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

		// Set the validation dataset
		bool			setValidationData(DataSet* dataset);

        private:

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
