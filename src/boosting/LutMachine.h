#ifndef _TORCH5SPRO_LUT_MACHINE_H_
#define _TORCH5SPRO_LUT_MACHINE_H_

#include "Machine.h"

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

	class LutMachine : public Machine
	{

	public:

		/// Constructor
		LutMachine();

		/// Destructor
		virtual ~LutMachine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new LutMachine(); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return LUT_MACHINE_ID; }

		///////////////////////////////////////////////////////////
		// Access functions

		void 			setParams(double min_, double max_, int n_bins_, double *lut_);

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// parameters of the machine
		double min, max;
		int n_bins;
		double *lut;
	};
}

#endif
