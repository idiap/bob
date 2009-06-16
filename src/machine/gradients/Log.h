#ifndef _TORCH5SPRO_LOG_MACHINE_H_
#define _TORCH5SPRO_LOG_MACHINE_H_

#include "GradientMachine.h"	// Log is a <GradientMachine>
#include "Machines.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Log:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Log : public GradientMachine
	{
	public:

		/// Constructor
		Log();

		Log(const int n_units_);

		/// Destructor
		virtual ~Log();

		///////////////////////////////////////////////////////////

		virtual bool 		forward(const DoubleTensor *input);

		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha);

		/// Constructs an empty Machine of this kind - overriden
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new Log(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return LOG_GRADIENT_MACHINE_ID; }

		///////////////////////////////////////////////////////////

	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool log_gradient_machine_registered = MachineManager::getInstance().add(new Log(), "Log");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
