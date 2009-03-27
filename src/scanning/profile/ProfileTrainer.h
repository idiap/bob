#ifndef _TORCHVISION_SCANNING_PROFILE_TRAINER_H_
#define _TORCHVISION_SCANNING_PROFILE_TRAINER_H_

#include "Trainer.h"		// <ProfileTrainer> is a <Trainer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ProfileTrainer:
	//	- trains ProfileMachine using ProfileDataSet objects:
	//		- one for training profile feature models (Trainer::setData)
	//		- one for selecting the best features and training the combined classifier
	//			(ProfileTrainer::setValidationData)
        //
        //      - PARAMETERS (name, type, default value, description):
        //		"FMinTAR"	float	0.85	"Minimum TAR for some profile feature model"
        //		"FMinTRR"	float	0.85	"Minimum TRR for some profile feature model"
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

        private:

                /////////////////////////////////////////////////////////////////
      		// Attributes

		DataSet*		m_validation_dataset;
	};
}

#endif
