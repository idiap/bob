#ifndef GRADIENT_MACHINE_INC
#define GRADIENT_MACHINE_INC

#include "Machine.h"

namespace Torch {

	class GradientMachine : public Machine
	{
	public:

		/// Constructor assuming input/output tensors of dimension 1
		GradientMachine(unsigned int input_size_, unsigned int output_size_, unsigned int n_parameters_ = 0);

		/// Destructor
		virtual ~GradientMachine();

		/// Forward the input tensor into the machine
		virtual bool 		forward(const DoubleTensor *input);
		virtual bool 		forward(int t, double *input, double *output);

		/// Backward the input tensor into the machine with alpha
		virtual bool 		backward(const DoubleTensor *input, const DoubleTensor *alpha);
		virtual bool 		backward(int t, double *input, double *beta, double *output, double* alpha);

	protected:

		///////////////////////////////////////////////////////////////
		/// Attributes

		///
		unsigned int input_size;

		///
		unsigned int output_size;

		/// Contains the derivative with respect to the input
		DoubleTensor		*beta;

		/// Derivative of the parameters of the machine
		double			*der_parameters;
	};

}

#endif
