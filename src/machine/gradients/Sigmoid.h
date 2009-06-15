#ifndef _TORCH5SPRO_SIGMOID_MACHINE_H_
#define _TORCH5SPRO_SIGMOID_MACHINE_H_

#include "GradientMachine.h"	// Exp is a <GradientMachine>
#include "Machines.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Sigmoid:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Sigmoid : public GradientMachine
	{
	public:

		/// Constructor
		Sigmoid();

		Sigmoid(const int n_units_);

		/// Destructor
		virtual ~Sigmoid();

		///////////////////////////////////////////////////////////

		///
		virtual bool 		forward(const DoubleTensor *input);

		///
		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha);

		/// Constructs an empty Machine of this kind - overriden
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new Sigmoid(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return SIGMOID_GRADIENT_MACHINE_ID; }

		///////////////////////////////////////////////////////////

	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool sigmoid_gradient_machine_registered = MachineManager::getInstance().add(new Sigmoid(), "Sigmoid");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
