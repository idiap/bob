#include "GradientMachine.h"
#include "Tensor.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

GradientMachine::GradientMachine(unsigned int input_size_, unsigned int output_size_, unsigned int n_parameters_)
	: 	Machine(n_parameters_)
{
	input_size = input_size_;
	output_size = output_size_;

	if(n_parameters == 0) der_parameters = NULL;
	else der_parameters = new double [n_parameters];
}

bool GradientMachine::forward(const DoubleTensor *input)
{
	if(input->nDimension() == 1)
	{
	   	if(input->size(0) != input_size)
		{
			warning("GradientMachine::forward() incorrect size of the input tensor (%d != %d).", input->size(0), input_size);
			return false;
		}

		// then forward the vector into the machine
		return forward(0, &input->get(0), &getOutput()->get(0));
	}
	else if(input->nDimension() == 2)
	{
		// then forward each row as a vector 
	}

	return false;
}

bool GradientMachine::forward(int t, double *input, double *output)
{
	return true;
}

bool GradientMachine::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
	return true;
}

bool GradientMachine::backward(int t, double *input, double *beta, double *output, double* alpha)
{
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Destructor

GradientMachine::~GradientMachine()
{
	if(der_parameters != NULL) delete [] der_parameters;
}

///////////////////////////////////////////////////////////////////////////

}

