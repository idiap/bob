#ifndef _TORCH5SPRO_REALLUT_MACHINE_H_
#define _TORCH5SPRO_REALLUT_MACHINE_H_

#include "machine/Machine.h"
#include "machine/Machines.h"

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

	class RealLutMachine : public Machine
	{

	public:

		/// Constructor
		RealLutMachine();

		/// Destructor
		virtual ~RealLutMachine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		// Loading/Saving the content from files (<em>not the options</em>) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine*	getAnInstance() const { return manage(new RealLutMachine()); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return REAL_LUT_MACHINE_ID; }

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
		bool verbose;
	};

	// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool real_lut_machine_registered = MachineManager::getInstance().add(
		new RealLutMachine(), "RealLutMachine");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
