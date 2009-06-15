#ifndef _TORCH5SPRO_EXP_MACHINE_H_
#define _TORCH5SPRO_EXP_MACHINE_H_

#include "GradientMachine.h"	// Exp is a <GradientMachine>
#include "Machines.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Exp:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Exp : public GradientMachine
	{
	public:

		/// Constructor
		Exp();

		Exp(const int n_units_);

		/// Destructor
		virtual ~Exp();

		///////////////////////////////////////////////////////////

		///
		virtual bool 		forward(const DoubleTensor *input);

		///
		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha);

		/// Constructs an empty Machine of this kind - overriden
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new Exp(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return EXP_GRADIENT_MACHINE_ID; }

		///////////////////////////////////////////////////////////

	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool exp_gradient_machine_registered = MachineManager::getInstance().add(new Exp(), "Exp");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
