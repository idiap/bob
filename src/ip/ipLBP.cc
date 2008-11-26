#include "ipLBP.h"
#include "Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipLBP::ipLBP(int P, int R)
	:	ipCore(),
		m_P(P), m_R(R),
		m_x(0), m_y(0),
		m_lbp(0),
		m_toAverage(false), m_addAvgBit(false), m_uniform(0), m_rot_invariant(false)
{
	addBOption("ToAverage", false, "compute the LBP code to the average");
	addBOption("AddAvgBit", false, "add to the LBP code the center (considering the average)");
	addBOption("Uniform", false, "uniform patterns (at most two bitwise 0-1 or 1-0 transitions)");
	addBOption("RotInvariant", false, "rotation invariant patterns");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipLBP::~ipLBP()
{
}

/////////////////////////////////////////////////////////////////////////
// called when some option was changed - overriden

void ipLBP::optionChanged(const char* name)
{
	// Just recompute the LBP parameters
	m_toAverage = getBOption("ToAverage");
	m_addAvgBit = getBOption("AddAvgBit");
	m_uniform = getBOption("Uniform");
	m_rot_invariant = getBOption("RotInvariant");
}

/////////////////////////////////////////////////////////////////////////
// Set the LBP location

bool ipLBP::setXY(int x, int y)
{
	if (	x >= m_R && x + m_R < m_inputSize.w &&
		y >= m_R && y + m_R < m_inputSize.h)
	{
		m_x = x;
		m_y = y;
		return true;
	}
	return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Set the radius value of the LBP operator

bool ipLBP::setR(int R)
{
	if (	R > 0 && R < m_inputSize.w / 2 && R < m_inputSize.h / 2)
	{
		m_R = R;
		return true;
	}
	return false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type - overriden

bool ipLBP::checkInput(const Tensor& input) const
{
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

bool ipLBP::allocateOutput(const Tensor& input)
{
	// Allocate the output if needed
	if (m_output == 0)
	{
	        m_n_outputs = 1;
                m_output = new Tensor*[m_n_outputs];
                m_output[0] = new IntTensor(1);
                const IntTensor* t_output = (IntTensor*)m_output[0];
                m_lbp = t_output->t->storage->data + t_output->t->storageOffset;
	}
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Bilinear interpolation

short ipLBP::bilinear_interpolation(const short* src, int stride_w, int stride_h, float x, float y)
{
	int xl = (int) floor(x);
	int yl = (int) floor(y);
	int xh = (int) ceil(x);
	int yh = (int) ceil(y);

	const float Il = src[xl * stride_w + yl * stride_h] + (x - xl) *
			(src[xh * stride_w + yl * stride_h] - src[xl * stride_w + yl * stride_h]);
	const float Ih = src[xl * stride_w + yh * stride_h] + (x - xl) *
			(src[xh * stride_w + yh * stride_h] - src[xl * stride_w + yh * stride_h]);

	return (short)(Il + (y - yl) * (Ih - Il) + 0.5f);
}

/////////////////////////////////////////////////////////////////////////

}
