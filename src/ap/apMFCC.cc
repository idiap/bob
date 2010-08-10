#include "ap/apMFCC.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

apMFCC::apMFCC()
	:	apCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

apMFCC::~apMFCC()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool apMFCC::checkInput(const Tensor& input) const
{
	// Accept only tensors of Torch::Float
	if (input.getDatatype() != Tensor::Float) return false;


	if (input.nDimension() != 1)
	{
		warning("apMFCC(): input dimension should be 1.");
		return false;
	}
	
	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool apMFCC::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 )
	{
		cleanup();
	
		if (input.nDimension() == 1)
		{
			print("apMFCC::allocateOutput() MFCC ...\n");

			int N = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(N);
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool apMFCC::processInput(const Tensor& input)
{
	const FloatTensor* t_input = (FloatTensor*)&input;

	m_output[0]->copy(t_input);

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

