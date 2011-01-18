#include "ip/ipHisto.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipHisto::ipHisto()
	:	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipHisto::~ipHisto()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipHisto::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 ||
		input.getDatatype() != Tensor::Short)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipHisto::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 2 ||
		m_output[0]->size(0) != 256 ||
		m_output[0]->size(1) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new IntTensor(256, input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipHisto::processInput(const Tensor& input)
{
	// Prepare the input and output (histogram) tensors
	const ShortTensor* t_input = (ShortTensor*)&input;
	IntTensor* t_output = (IntTensor*)m_output[0];
  t_output->fill(0);
	for (int x=0; x<input.size(1); ++x)
		for (int y = 0; y<input.size(0); ++y)
      for (int p = 0; p<input.size(2); ++p)
        ++(*t_output)((*t_input)(y, x, p), p);
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
