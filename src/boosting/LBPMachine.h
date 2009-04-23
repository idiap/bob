#ifndef _TORCH5SPRO_LBP_MACHINE_H_
#define _TORCH5SPRO_LBP_MACHINE_H_

#include "Machine.h"
#include "spCoreManager.h"
namespace Torch {


	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::LUTMachine:
	//      Process some input using a model (loaded from some file).
	//      The output is a DoubleTensor!
	//
	//      NB: The ouput should be allocated and deallocated by each Machine implementation!
	//
	//	EACH MACHINE SHOULD REGISTER
	//		==> MachineManager::GetInstance().add(new XXXMachine) <==
	//	TO THE MACHINEMANAGER CLASS!!!
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class LBPMachine : public Machine
	{

	public:

		/// Constructor
		LBPMachine();

		/// Destructor
		virtual ~LBPMachine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new LBPMachine(); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return LBP_MACHINE_ID; }

		///////////////////////////////////////////////////////////
		// Access functions

		void 			setParams(int n_bins_, double *lut_);

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// parameters of the machine
		//double min, max;
		int n_bins;
		double *lut;
	};


	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool lbp_machine_registered = MachineManager::getInstance().add(new LBPMachine(), "LBPMachine");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
