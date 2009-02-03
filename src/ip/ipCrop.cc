#include "ipCrop.h"
#include "Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipCrop::ipCrop()
	:	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipCrop::~ipCrop()
{
}

/////////////////////////////////////////////////////////////////////////
// Change the cropping area

bool ipCrop::setCropArea(int x, int y, int w, int h)
{
	if (	x >= 0 && y >= 0 && w > 0 && h > 0)
	{
		if (	m_cropArea.x != x ||
			m_cropArea.y != y ||
			m_cropArea.w != w ||
			m_cropArea.h != h)
		{
			// Delete the old tensors (if any)
			cleanup();
			m_cropArea.x = x;
			m_cropArea.y = y;
			m_cropArea.w = w;
			m_cropArea.h = h;
		}
		return true;
	}
	return false;
}

bool ipCrop::setCropArea(const sRect2D& area)
{
	return setCropArea(area.x, area.y, area.w, area.h);
}

//////////////////////////////////////////////////////////////////////////
// Retrieve the cropping area

const sRect2D& ipCrop::getCropArea() const
{
	return m_cropArea;
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
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != m_cropArea.h ||
		m_output[0]->size(1) != m_cropArea.w ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(m_cropArea.h, m_cropArea.w, input.size(2));
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipCrop::processInput(const Tensor& input)
{
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	//const short* src = t_input->t->storage->data + t_input->t->storageOffset;
	//short* dst = t_output->t->storage->data + t_output->t->storageOffset;

	//const int in_stride_h = t_input->t->stride[0];	// height
	//const int in_stride_w = t_input->t->stride[1];	// width
	//const int in_stride_p = t_input->t->stride[2];	// no planes

	//const int out_stride_h = t_output->t->stride[0];	// height
	//const int out_stride_w = t_output->t->stride[1];	// width
	//const int out_stride_p = t_output->t->stride[2];	// no planes

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]

	const int n_planes = input.size(2);

	// Cosmin: To be optimized!
	//
	// Seb: Yes, this can be optimized later using 2 times the Tensor:narrow
	// function assuming that the resulting crop tensor will not be
	// modified (boolean option in the constructor)
	for (int p = 0; p < n_planes; p ++)
	{
		for (int y = 0; y < m_cropArea.h; y ++)
		{
			const int in_y = y + m_cropArea.y;
			for (int x = 0; x < m_cropArea.w; x ++)
			{
				const int in_x = x + m_cropArea.x;
				t_output->set(y, x, p, t_input->get(in_y, in_x, p));
			}
		}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
