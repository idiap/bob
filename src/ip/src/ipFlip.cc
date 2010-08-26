#include "ip/ipFlip.h"
#include "core/Tensor.h"
#include <iostream>

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipFlip::ipFlip() : ipCore()
{
	addBOption("vertical", false, "direction of the flipping (default vertical)");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipFlip::~ipFlip()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipFlip::checkInput(const Tensor& input) const
{
	// Torch::Image type
	if (input.getDatatype() != Tensor::Short &&
		input.nDimension() != 3)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipFlip::allocateOutput(const Tensor& input)
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

static int getMirror(int val, int max)
{
	return (max - val);
}

static int getNormal(int val, int max)
{
	return (val);
}

bool ipFlip::processInput(const Tensor& input)
{
	const bool vertical = getBOption("vertical");

	// downcast to "image" and note down size and type
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	// source and destination in memory
	const short* src = (const short*)t_input->dataR();
	short* dst = (short*)t_output->dataW();

	// the indexing of the data
	const int stride_h = t_input->stride(0);        // height
	const int stride_w = t_input->stride(1);        // width
	const int in_x_max = input.size(0);
	const int in_y_max = input.size(1);

	// difference between 2d and 3d
	int stride_p;
	int n_planes;
	if (3 == t_input->nDimension())
	{
		stride_p = t_input->stride(2);
		n_planes = input.size(2);
	}
	else
	{
		stride_p = 0;
		n_planes = 1;
	}

	// create two function pointers to used for fliping
	// since the operations are so simulare
	int (* fp_x) (int, int);
	int (* fp_y) (int, int);

	if (vertical == true)
	{
		// flip over horizontal axis
		fp_x = getNormal;
		fp_y = getMirror;
	}
	else
	{
		// flip over vertical axis
		fp_x = getMirror;
		fp_y = getNormal;
	}


	// main algorithm
	for (int p = 0; p < n_planes; ++p) {

		for (int in_x = 0; in_x < in_x_max; ++in_x) {

			for (int in_y = 0; in_y < in_y_max; ++in_y) {

				// normal
				int in_index = in_x * stride_h + in_y * stride_w + p * stride_p;

				// flip
				int out_x = fp_x(in_x, in_x_max - 1);
				int out_y = fp_y(in_y, in_y_max - 1);
				int out_index = out_x * stride_h + out_y * stride_w + p * stride_p;

				dst[out_index] = src[in_index];
			}
		}
	}

	// OK
  t_output->resetFromData();
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
