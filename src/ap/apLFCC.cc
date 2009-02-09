#include "apLFCC.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

apLFCC::apLFCC()
	:	apCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

apLFCC::~apLFCC()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool apLFCC::checkInput(const Tensor& input) const
{
	// Accept only tensors of Torch::Float
	if (input.getDatatype() != Tensor::Float) return false;


	if (input.nDimension() != 1)
	{
		warning("apLFCC(): input dimension should be 1.");
		return false;
	}
	
	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool apLFCC::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 )
	{
		cleanup();
	
		if (input.nDimension() == 1)
		{
			print("apLFCC::allocateOutput() LFCC ...\n");

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

bool apLFCC::processInput(const Tensor& input)
{
	const FloatTensor* t_input = (FloatTensor*)&input;

	m_output[0]->copy(t_input);

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

