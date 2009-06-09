#ifndef _TORCHVISION_SCANNING_PROFILE_MACHINE_H_
#define _TORCHVISION_SCANNING_PROFILE_MACHINE_H_

#include "Classifier.h"			// <ProfileMachine> is a <Classifier>
#include "LRMachine.h"
#include "Profile.h"

namespace Torch
{
	#define PROFILE_MACHINE_ID	10003

	/////////////////////////////////////////////////////////////////////////
	// Torch::ProfileMachine:
	//	- implements a combination of small machines for each profile feature
	//
	//      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ProfileMachine : public Torch::Classifier
	{
	public:

		// Constructor
		ProfileMachine();

		// Destructor
		virtual ~ProfileMachine();

		// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine* 	getAnInstance() const { return manage(new ProfileMachine); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return PROFILE_MACHINE_ID; }

		// Loading/Saving the content from files (\emph{not the options})
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		// Access functions
		LRMachine&		getFModel(int f) { return m_fmodels[f]; }
		const LRMachine&	getFModel(int f) const { return m_fmodels[f]; }
		LRMachine&		getCModel() { return m_cmodel; }
		const LRMachine&	getCModel() const  { return m_cmodel; }

		/////////////////////////////////////////////////////////////////

        private:

                /////////////////////////////////////////////////////////////////
                // Attributes

		double*			m_poutput;		// Direct access to the machine's output
		DoubleTensor		m_foutputs;		// Store outputs from profile feature models

		LRMachine*		m_fmodels;		// Models for each feature
		LRMachine		m_cmodel;		// Combined model

		Profile			m_profile;		// Buffered sample for easily access the features
	};
}

#endif
