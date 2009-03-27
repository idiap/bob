#include "ipCrop.h"
#include "Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipCrop::ipCrop()
	:	ipCore()
{
	addIOption("x", 0, "Ox coordinate of the top left corner of the cropping area");
	addIOption("y", 0, "Oy coordinate of the top left corner of the cropping area");
	addIOption("w", 0, "desired width of the cropped image");
	addIOption("h", 0, "desired height of the cropped image");
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

	// Prepare direct access to data
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const short* src = (const short*)t_input->dataR();
	short* dst = (short*)t_output->dataW();

	const int src_stride_h = t_input->t->stride[0];	// height
	const int src_stride_w = t_input->t->stride[1];	// width
	const int src_stride_p = t_input->t->stride[2];	// no planes

	const int dst_stride_h = t_output->t->stride[0];// height
	const int dst_stride_w = t_output->t->stride[1];// width
	const int dst_stride_p = t_output->t->stride[2];// no planes

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	//const int src_height = t_input->size(0);
	//const int src_width = t_input->size(1);
	const int n_planes = input.size(2);

	// Cropping - just copy pixels in the given range
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_plane = &src[p * src_stride_p];
		short* dst_plane = &dst[p * dst_stride_p];

		for (int y = 0; y < crop_h; y ++)
		{
			const short* src_row = &src_plane[(crop_y + y) * src_stride_h + crop_x * src_stride_w];
			short* dst_row = &dst_plane[y * dst_stride_h];

			for (int x = 0; x < crop_w; x ++, src_row += src_stride_w, dst_row += dst_stride_w)
			{
				*dst_row = *src_row;
			}
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
