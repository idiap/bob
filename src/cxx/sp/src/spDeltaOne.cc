#include "sp/spDeltaOne.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

spDeltaOne::spDeltaOne(int n_)
	:	spCore()
{
   	n = n_;
}

/////////////////////////////////////////////////////////////////////////
// Destructor

spDeltaOne::~spDeltaOne()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool spDeltaOne::checkInput(const Tensor& input) const
{
	// Accept only tensors of Torch::Double
	if (input.getDatatype() != Tensor::Double) return false;

	if (input.nDimension() == 1) return true;
	else
	{
	   	Torch::error("spDeltaOne::checkInput() doesn't handle more than 1D ...\n");

		return false;
	}
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool spDeltaOne::allocateOutput(const Tensor& input)
{
	if ( m_output == 0 )
	{
		cleanup();
	
		if (input.nDimension() == 1)
		{
			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new DoubleTensor(1);

			return true;
		}
		else
		{
	   		Torch::error("spDeltaOne::allocateOutput() doesn't handle more than 1D ...\n");

			return false;
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool spDeltaOne::processInput(const Tensor& input)
{
	DoubleTensor* t_input = (DoubleTensor*) &input;
	DoubleTensor* t_output = (DoubleTensor*) m_output[0];

	if (input.nDimension() == 1)
	{
	   	(*t_output)(0) = (*t_input)((long) n); 

		return true;
	}

	return false;
}

/////////////////////////////////////////////////////////////////////////

}

