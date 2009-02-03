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
		m_lut_RI(0),
		m_lut_U2(0),
		m_lut_U2RI(0),
		m_lut_addAvgBit(0),
		m_lut_normal(0),
		m_crt_lut(0),
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
	delete[] m_lut_RI;
	delete[] m_lut_U2;
	delete[] m_lut_U2RI;
	delete[] m_lut_addAvgBit;
	delete[] m_lut_normal;
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

	// Set the current conversion table
	if (m_rot_invariant == true)
	{
		if (m_uniform == true)
		{
			m_crt_lut = m_lut_U2RI;
		}
		else
		{
			m_crt_lut = m_lut_RI;
		}
	}
	else
	{
		if (m_uniform == true)
		{
			m_crt_lut = m_lut_U2;
		}
		else
		{
			if (m_addAvgBit == true && m_toAverage == true)
			{
				m_crt_lut = m_lut_addAvgBit;
			}
			else
			{
				m_crt_lut = m_lut_normal;
			}
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Set the LBP location

bool ipLBP::setXY(int x, int y)
{
	if (	x >= m_R &&
		y >= m_R)
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
	if (	R > 0)
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
	if (	input.nDimension() != 2 &&
		input.nDimension() != 3)
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

char ipLBP::bilinear_interpolation(const char* src, int stride_w, int stride_h, float x, float y)
{
	int xl = (int) floor(x);
	int yl = (int) floor(y);
	int xh = (int) ceil(x);
	int yh = (int) ceil(y);

	const float Il = src[xl * stride_w + yl * stride_h] + (x - xl) *
			(src[xh * stride_w + yl * stride_h] - src[xl * stride_w + yl * stride_h]);
	const float Ih = src[xl * stride_w + yh * stride_h] + (x - xl) *
			(src[xh * stride_w + yh * stride_h] - src[xl * stride_w + yh * stride_h]);

	return (char)(Il + (y - yl) * (Ih - Il) + 0.5f);
}

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

int ipLBP::bilinear_interpolation(const int* src, int stride_w, int stride_h, float x, float y)
{
	int xl = (int) floor(x);
	int yl = (int) floor(y);
	int xh = (int) ceil(x);
	int yh = (int) ceil(y);

	const float Il = src[xl * stride_w + yl * stride_h] + (x - xl) *
			(src[xh * stride_w + yl * stride_h] - src[xl * stride_w + yl * stride_h]);
	const float Ih = src[xl * stride_w + yh * stride_h] + (x - xl) *
			(src[xh * stride_w + yh * stride_h] - src[xl * stride_w + yh * stride_h]);

	return (int)(Il + (y - yl) * (Ih - Il) + 0.5f);
}

long ipLBP::bilinear_interpolation(const long* src, int stride_w, int stride_h, float x, float y)
{
	int xl = (int) floor(x);
	int yl = (int) floor(y);
	int xh = (int) ceil(x);
	int yh = (int) ceil(y);

	const float Il = src[xl * stride_w + yl * stride_h] + (x - xl) *
			(src[xh * stride_w + yl * stride_h] - src[xl * stride_w + yl * stride_h]);
	const float Ih = src[xl * stride_w + yh * stride_h] + (x - xl) *
			(src[xh * stride_w + yh * stride_h] - src[xl * stride_w + yh * stride_h]);

	return (long)(Il + (y - yl) * (Ih - Il) + 0.5f);
}

float ipLBP::bilinear_interpolation(const float* src, int stride_w, int stride_h, float x, float y)
{
	int xl = (int) floor(x);
	int yl = (int) floor(y);
	int xh = (int) ceil(x);
	int yh = (int) ceil(y);

	const float Il = src[xl * stride_w + yl * stride_h] + (x - xl) *
			(src[xh * stride_w + yl * stride_h] - src[xl * stride_w + yl * stride_h]);
	const float Ih = src[xl * stride_w + yh * stride_h] + (x - xl) *
			(src[xh * stride_w + yh * stride_h] - src[xl * stride_w + yh * stride_h]);

	return Il + (y - yl) * (Ih - Il);
}

double ipLBP::bilinear_interpolation(const double* src, int stride_w, int stride_h, float x, float y)
{
	int xl = (int) floor(x);
	int yl = (int) floor(y);
	int xh = (int) ceil(x);
	int yh = (int) ceil(y);

	const double Il = src[xl * stride_w + yl * stride_h] + (x - xl) *
			(src[xh * stride_w + yl * stride_h] - src[xl * stride_w + yl * stride_h]);
	const double Ih = src[xl * stride_w + yh * stride_h] + (x - xl) *
			(src[xh * stride_w + yh * stride_h] - src[xl * stride_w + yh * stride_h]);

	return Il + (y - yl) * (Ih - Il);
}

/////////////////////////////////////////////////////////////////////////

}
