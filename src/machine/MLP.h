#ifndef _TORCH5SPRO_MLP_MACHINE_H_
#define _TORCH5SPRO_MLP_MACHINE_H_

#include "GradientMachine.h"	// MLP is a <GradientMachine>
#include "Machines.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::MLP:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class MLP : public GradientMachine
	{
	public:

		/// Constructor
		MLP();

		MLP(const int n_inputs_, const int n_hidden_units_, const int n_outputs_);

		/// Destructor
		virtual ~MLP();

		///////////////////////////////////////////////////////////

		///
		virtual bool 		prepare();

		///
		virtual bool 		shuffle();

		///
		virtual bool 		Ginit();
		
		///
		virtual bool 		Gupdate(double learning_rate);
		
		///
		virtual bool 		forward(const DoubleTensor *input);

		///
		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha);

		/// Constructs an empty Machine of this kind - overriden
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new MLP(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return MLP_GRADIENT_MACHINE_ID; }

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		///////////////////////////////////////////////////////////
	protected:

		int n_gm;
		GradientMachine **gm;
	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool mlp_gradient_machine_registered = MachineManager::getInstance().add(new MLP(), "MLP");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
