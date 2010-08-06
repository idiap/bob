#ifndef _TORCHVISION_SCANNING_CONTEXT_TRAINER_H_
#define _TORCHVISION_SCANNING_CONTEXT_TRAINER_H_

#include "core/Trainer.h"		// <ContextTrainer> is a <Trainer>

namespace Torch
{
	class ContextMachine;
	class ContextDataSet;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ContextTrainer:
	//	- trains ContextMachine using two ContextDataSets:
	//		- one for training and one for validation
        //
        //      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ContextTrainer : public Torch::Trainer
	{
	public:

		// Constructor
		ContextTrainer();

		// Destructor
		virtual ~ContextTrainer();

		// Train the given machine on the given dataset
		virtual bool 		train();

		// Set the validation dataset
		bool			setValidationData(DataSet* dataset);

		// Test the Context machine
		static void		test(ContextMachine* machine, ContextDataSet* samples,
						double& TAR, double& FAR, double& HTER);

        private:

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
