#ifndef _TORCHVISION_SCANNING_PROFILE_TRAINER_H_
#define _TORCHVISION_SCANNING_PROFILE_TRAINER_H_

#include "Trainer.h"		// <ProfileTrainer> is a <Trainer>

namespace Torch
{
	class ProfileMachine;
	class ProfileDataSet;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ProfileTrainer:
	//	- trains ProfileMachine using two ProfileDataSets:
	//		- one for training and one for validation
        //
        //      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ProfileTrainer : public Torch::Trainer
	{
	public:

		// Constructor
		ProfileTrainer();

		// Destructor
		virtual ~ProfileTrainer();

		// Train the given machine on the given dataset
		virtual bool 		train();

		// Set the validation dataset
		bool			setValidationData(DataSet* dataset);

		// Test the Profile machine (returns the detection rate in percentages)
		static double		test(ProfileMachine* machine, ProfileDataSet* samples);

		// Test the Profile machine (returns the TAR and FAR and FA)
		static void		test(ProfileMachine* machine, ProfileDataSet* samples,
						double& tar, double& far, long& fa);

        private:

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
