#ifndef _TORCH5SPRO_TANH_MACHINE_H_
#define _TORCH5SPRO_TANH_MACHINE_H_

#include "GradientMachine.h"	// Tanh is a <GradientMachine>
#include "Machines.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Tanh:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Tanh : public GradientMachine
	{
	public:

		/// Constructor
		Tanh();

		Tanh(const int n_units_);

		/// Destructor
		virtual ~Tanh();

		///////////////////////////////////////////////////////////

		///
		virtual bool 		forward(const DoubleTensor *input);

		///
		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha);

		/// Constructs an empty Machine of this kind - overriden
		// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new Tanh(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return TANH_GRADIENT_MACHINE_ID; }

		///////////////////////////////////////////////////////////

	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool tanh_gradient_machine_registered = MachineManager::getInstance().add(new Tanh(), "Tanh");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
