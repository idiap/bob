#ifndef _TORCH5SPRO_INTLUT_MACHINE_H_
#define _TORCH5SPRO_INTLUT_MACHINE_H_

#include "core/Machine.h"
#include "Machines.h"

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

	class IntLutMachine : public Machine
	{

	public:

		/// Constructor
		IntLutMachine();

		/// Destructor
		virtual ~IntLutMachine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind
		/// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine*	getAnInstance() const { return manage(new IntLutMachine()); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return INT_LUT_MACHINE_ID; }

		void 			setParams(int n_bins_, double *lut_);

		///////////////////////////////////////////////////////////
		// Access functions

		int			getLUTSize() const { return n_bins; }
		double*			getLUT()  { return lut; }

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
        const bool intlut_machine_registered = MachineManager::getInstance().add(
		new IntLutMachine(), "IntLutMachine");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
