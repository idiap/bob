#include "ip/ipCrop.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipCrop::ipCrop()
	:	ipCore()
{
	addIOption("x", 0, "Ox coordinate of the top left corner of the cropping area");
	addIOption("y", 0, "Oy coordinate of the top left corner of the cropping area");
	addIOption("w", 1, "desired width of the cropped image");
	addIOption("h", 1, "desired height of the cropped image");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipCrop::~ipCrop()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipCrop::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 || input.getDatatype() != Tensor::Short) {
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipCrop::allocateOutput(const Tensor& input)
{
	const int crop_x = getIOption("x");
	const int crop_y = getIOption("y");
	const int crop_w = getIOption("w");
	const int crop_h = getIOption("h");

	// Check parameters
	if (	crop_x < 0 || crop_y < 0 || crop_w < 0 || crop_h < 0 ||
		crop_x + crop_w > input.size(1) ||
		crop_y + crop_h > input.size(0))
	{
		return false;
	}

	// Allocate output if required
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != crop_h ||
		m_output[0]->size(1) != crop_w ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(crop_h, crop_w, input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipCrop::processInput(const Tensor& input)
{
	// Get parameters
	const int crop_x = getIOption("x");
	const int crop_y = getIOption("y");
	const int crop_w = getIOption("w");
	const int crop_h = getIOption("h");
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

  for (int cx=0; cx<crop_w; ++cx) {
    for (int cy=0; cy<crop_h; ++cy) {
      for (int cp=0; cp<t_input->size(2); ++cp)
        (*t_output)(cy, cx, cp) = (*t_input)(cy+crop_y, cx+crop_x, cp);
    }
  }

	return true;
}

/////////////////////////////////////////////////////////////////////////

}
