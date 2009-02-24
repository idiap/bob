#ifndef _TORCHVISION_SCANNING_PROFILE_MACHINE_H_
#define _TORCHVISION_SCANNING_PROFILE_MACHINE_H_

#include "Machine.h"			// <ProfileMachine> is a <Machine>

namespace Torch
{
namespace Profile
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Profile::ProfileMachine:
	//	- implements a combination of small machines for each profile feature
	//
	//      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ProfileMachine : public Machine
	{
	public:

		// Constructor
		ProfileMachine();

		// Destructor
		virtual ~ProfileMachine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		/// Constructs an empty Machine of this kind
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const;

		// Get the ID specific to each Machine
		virtual int		getID() const;

		/// Loading/Saving the content from files (\emph{not the options})
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/////////////////////////////////////////////////////////////////

        private:

                /////////////////////////////////////////////////////////////////
                // Attributes

		//
	};
}
}

#endif
