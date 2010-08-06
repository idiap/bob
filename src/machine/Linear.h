#ifndef _TORCH5SPRO_LINEAR_MACHINE_H_
#define _TORCH5SPRO_LINEAR_MACHINE_H_

#include "GradientMachine.h"	// Linear is a <GradientMachine>
#include "Machines.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Sigmoid:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Linear : public GradientMachine
	{
	public:

		/// Constructor
		Linear();

		Linear(const int n_inputs_, const int n_outputs_);

		/// Destructor
		virtual ~Linear();

		///////////////////////////////////////////////////////////

		///
		virtual bool 		resize(const int n_inputs_, const int n_outputs_, const int n_parameters_ = 0);

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
		virtual Machine*	getAnInstance() const { return new Linear(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return LINEAR_GRADIENT_MACHINE_ID; }

		///////////////////////////////////////////////////////////

	protected:
		double *weights;
		double *bias;
		double *der_weights;
		double *der_bias;
		double *delta_parameters;
	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool linear_gradient_machine_registered = MachineManager::getInstance().add(new Linear(), "Linear");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
