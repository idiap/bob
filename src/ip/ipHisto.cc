#include "ipHisto.h"
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

	const short* src = t_input->t->storage->data + t_input->t->storageOffset;
	int* dst = t_output->t->storage->data + t_output->t->storageOffset;

	const int in_stride_h = t_input->t->stride[0];	// height
	const int in_stride_w = t_input->t->stride[1];	// width
	const int in_stride_p = t_input->t->stride[2];	// no planes

	const int out_stride_b = t_output->t->stride[0];// bin index
	const int out_stride_p = t_output->t->stride[1];// no planes

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	const int height = input.size(0);
	const int width = input.size(1);
	const int n_planes = input.size(2);

	// Clear the histogram
	for (int p = 0; p < n_planes; p ++)
	{
		int* histo_plane = &dst[p * out_stride_p];
		for (int b = 0; b < 256; b ++, histo_plane += out_stride_b)
		{
			*histo_plane = 0;
		}
	}

	// Compute the histogram
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_plane = &src[p* in_stride_p];
		int* histo_plane = &dst[p * out_stride_p];

		for (int y = 0; y < height; y ++)
		{
			const short* src_row = &src_plane[y * in_stride_h];
			for (int x = 0; x < width; x ++, src_row += in_stride_w)
			{
				histo_plane[*(src_row) * out_stride_b] ++;
			}
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
