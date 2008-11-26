#include "ipIntegralImage.h"
#include "Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipIntegralImage::ipIntegralImage()
	:	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipIntegralImage::~ipIntegralImage()
{
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipIntegralImage::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 ||
		input.getDatatype() != Tensor::Short)
	{
		return false;
	}

	// Accept only tensors having the set image size
	if (	input.size(0) != m_inputSize.h ||
		input.size(1) != m_inputSize.w)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipIntegralImage::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != m_inputSize.h ||
		m_output[0]->size(1) != m_inputSize.w ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new IntTensor(m_inputSize.h, m_inputSize.w, input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipIntegralImage::processInput(const Tensor& input)
{
	const ShortTensor* t_input = (ShortTensor*)&input;
	IntTensor* t_output = (IntTensor*)m_output[0];

	// Prepare the arrays to work with
	const short* src = t_input->t->storage->data + t_input->t->storageOffset;
	int* dst = t_output->t->storage->data + t_output->t->storageOffset;

	const int stride_h = t_input->t->stride[0];	// height
	const int stride_w = t_input->t->stride[1];	// width
	const int stride_p = t_input->t->stride[2];	// no planes

	const int width = m_inputSize.w;
	const int height = m_inputSize.h;
	const int n_planes = input.size(2);

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	// For each color channel (plane)
	for (int p = 0; p < n_planes; p ++)
	{
		const int dp = p * stride_p;

		// First line (y == 0)
		dst[dp] = src[dp];
		for (int x = 1; x < width; x ++)
		{
			const int baseindex = x * stride_w + dp;
			dst[baseindex] = dst[baseindex - stride_w] + src[baseindex];
		}

		// The next lines depend only on the last one
		for (int y = 1; y < height; y ++)
		{
			int baseindex = y * stride_h + dp;
			int line = src[baseindex];

			// first element of line y
			dst[baseindex] = dst[baseindex - stride_h] + line;
			baseindex += stride_w;

			// the rest elements of line y
			for (int x = 1; x < width; x ++, baseindex += stride_w)
			{
				line += src[baseindex];
				dst[baseindex] = dst[baseindex - stride_h] + line;
			}
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
