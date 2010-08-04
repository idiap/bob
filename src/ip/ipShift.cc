#include "ipShift.h"
#include "core/Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipShift::ipShift()
	:	ipCore()
{
	addIOption("shiftx", 0, "variation on Ox axis");
	addIOption("shifty", 0, "variation on Oy axis");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipShift::~ipShift()
{
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipShift::checkInput(const Tensor& input) const
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

bool ipShift::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(1) != input.size(1) ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipShift::processInput(const Tensor& input)
{
	const int dx = getIOption("shiftx");
	const int dy = getIOption("shifty");

	// Check the variation against input size
	if (	dx < -input.size(1) || dx > input.size(1) ||
		dy < -input.size(0) || dy > input.size(0))
	{
		return false;
	}

	// Prepare pointers to access pixels
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const short* src = (const short*)t_input->dataR();
	short* dst = (short*)t_output->dataW();

	const int stride_h = t_input->t->stride[0];	// height
	const int stride_w = t_input->t->stride[1];	// width
	const int stride_p = t_input->t->stride[2];	// no planes

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	const int width = input.size(1);
	const int height = input.size(0);
	const int n_planes = input.size(2);

	// Fill the result image with black
	t_output->fill(0);

	// Compute the range of valid pixel positions in the shifted image
	const int start_x = getInRange(dx, 0, width - 1);
	const int start_y = getInRange(dy, 0, height - 1);
	const int stop_x = getInRange(width + dx, 0, width - 1);
	const int stop_y = getInRange(height + dy, 0, height - 1);
	const int dindex = dy * stride_h + dx * stride_w;

	// Shift each plane ...
	for (int p = 0; p < n_planes; p ++)
	{
		//	input: 	[y * stride_h + x * stride_w + p * stride_p]
		//		->>>
		//	output: [(y + dy) * stride_h + (x + dx) * stride_w + p * stride_p])
		const short* src_plane = &src[p * stride_p];
		short* dst_plane = &dst[p * stride_p];

		for (int y = start_y; y < stop_y; y ++)
		{
			const int index_row = y * stride_h + start_x * stride_w;
			const short* src_row = &src_plane[index_row - dindex];
			short* dst_row = &dst_plane[index_row];

			for (int x = start_x; x < stop_x; x ++, src_row += stride_w, dst_row += stride_w)
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
