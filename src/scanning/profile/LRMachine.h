#ifndef _TORCHVISION_SCANNING_PROFILE_LR_MACHINE_H_
#define _TORCHVISION_SCANNING_PROFILE_LR_MACHINE_H_

#include "Machine.h"			// <LRMachine> is a <Machine>

namespace Torch
{
namespace Profile
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Profile::LRMachine:
	//	- implements Logistic Regression Linear Discriminant Analysis (generic,
	//		but used with profiling scanning)
	//
	//      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class LRMachine : public Machine
	{
	public:

		// Constructor
		LRMachine();

		// Destructor
		virtual ~LRMachine();

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
