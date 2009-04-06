#include "ipLBP.h"
#include "Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipLBP::ipLBP(int P, int R)
	:	ipCore(),
		m_P(P), m_R(R),
		m_x(0), m_y(0),
		m_input_w(0), m_input_h(0), m_input_stride_w(0), m_input_stride_h(0),
		m_ii_tl(0), m_ii_tr(0), m_ii_bl(0), m_ii_br(0), m_ii_cell_size(0),
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

	for (int i = 0; i < m_modelSize.size[1]; i ++)
	{
		delete[] m_ii_tl[i];
		delete[] m_ii_tr[i];
		delete[] m_ii_bl[i];
		delete[] m_ii_br[i];
		delete[] m_ii_cell_size[i];
	}
	delete[] m_ii_tl;
	delete[] m_ii_tr;
	delete[] m_ii_bl;
	delete[] m_ii_br;
	delete[] m_ii_cell_size;
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
/// Change the region of the input tensor to process - overriden

void ipLBP::setRegion(const TensorRegion& region)
{
	const bool changed = 	m_region.size[0] != region.size[0] ||
				m_region.size[1] != region.size[1];

	ipCore::setRegion(region);
	if (changed == true)
	{
		updateIntegralFactors();
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Change the model size (if used with some machine) - overriden

void ipLBP::setModelSize(const TensorSize& modelSize)
{
	const bool changed = 	m_modelSize.size[0] != modelSize.size[0] ||
				m_modelSize.size[1] != modelSize.size[1];

	if (changed == true)
	{
		// Delete old indexes
		for (int i = 0; i < m_modelSize.size[1]; i ++)
		{
			delete[] m_ii_tl[i];
			delete[] m_ii_tr[i];
			delete[] m_ii_bl[i];
			delete[] m_ii_br[i];
			delete[] m_ii_cell_size[i];
		}
		delete[] m_ii_tl;
		delete[] m_ii_tr;
		delete[] m_ii_bl;
		delete[] m_ii_br;
		delete[] m_ii_cell_size;
	}

	ipCore::setModelSize(modelSize);

	if (changed == true)
	{
		const int model_w = m_modelSize.size[1];
		const int model_h = m_modelSize.size[0];

		// Allocate new indexes
		m_ii_tl = new int*[model_w];
		m_ii_tr = new int*[model_w];
		m_ii_bl = new int*[model_w];
		m_ii_br = new int*[model_w];
		m_ii_cell_size = new int*[model_w];

		for (int i = 0; i < model_w; i ++)
		{
			m_ii_tl[i] = new int[model_h];
			m_ii_tr[i] = new int[model_h];
			m_ii_bl[i] = new int[model_h];
			m_ii_br[i] = new int[model_h];
			m_ii_cell_size[i] = new int[model_h];
		}

		updateIntegralFactors();
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the scalling factors needed to interpolate using integral images

void ipLBP::updateIntegralFactors()
{
	const int model_w = m_modelSize.size[1];
	const int model_h = m_modelSize.size[0];

	if (	m_input_w == 0 || m_input_h == 0 ||
		model_w == 0 || model_h == 0)
	{
		return;
	}

	const int sw_w = m_region.size[1];
	const int sw_h = m_region.size[0];

	// Scalling factors
	const double inv_model_w = 1.0 / (model_w + 0.0);
        const double inv_model_h = 1.0 / (model_h + 0.0);
	const double scale_w = 0.5 * (sw_w + 0.0) / (model_w + 0.0);
        const double scale_h = 0.5 * (sw_h + 0.0) / (model_h + 0.0);

	// Compute the new indexes
	for (int i = 0; i < model_w; i ++)
	{
		for (int j = 0; j < model_h; j ++)
		{
                        const double x_in_sw = (i + 0.5) * inv_model_w * sw_w;
                        const double y_in_sw = (j + 0.5) * inv_model_h * sw_h;

                        const double min_x_in_sw = x_in_sw - scale_w - 0.5;
                        const double max_x_in_sw = x_in_sw + scale_w + 0.5;
                        const double min_y_in_sw = y_in_sw - scale_h - 0.5;
                        const double max_y_in_sw = y_in_sw + scale_h + 0.5;

                        const int l = getInRange((int)(min_x_in_sw), 0, sw_w - 1);
                        const int r = getInRange((int)(max_x_in_sw + 0.5), 0, sw_w - 1);
                        const int t = getInRange((int)(min_y_in_sw), 0, sw_h - 1);
                        const int b = getInRange((int)(max_y_in_sw + 0.5), 0, sw_h - 1);

                        m_ii_tl[i][j] = t * m_input_stride_h + l * m_input_stride_w;
			m_ii_tr[i][j] = t * m_input_stride_h + r * m_input_stride_w;
			m_ii_bl[i][j] = b * m_input_stride_h + l * m_input_stride_w;
                        m_ii_br[i][j] = b * m_input_stride_h + r * m_input_stride_w;
                        m_ii_cell_size[i][j] = (b - t) * (r - l);
                }
	}
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

	// If the input tensor size was modified, update the scalling factors
	if (	input.size(0) != m_input_h ||
		input.size(1) != m_input_w)
	{
		m_input_h = input.size(0);
		m_input_w = input.size(1);

		// Compute the strides of the input tensor
		switch (input.getDatatype())
		{
		case Tensor::Char:
			m_input_stride_h = ((const CharTensor*)&input)->t->stride[0];
			m_input_stride_w = ((const CharTensor*)&input)->t->stride[1];
			break;

		case Tensor::Short:
			m_input_stride_h = ((const ShortTensor*)&input)->t->stride[0];
			m_input_stride_w = ((const ShortTensor*)&input)->t->stride[1];
			break;

		case Tensor::Int:
			m_input_stride_h = ((const IntTensor*)&input)->t->stride[0];
			m_input_stride_w = ((const IntTensor*)&input)->t->stride[1];
			break;

		case Tensor::Long:
			m_input_stride_h = ((const LongTensor*)&input)->t->stride[0];
			m_input_stride_w = ((const LongTensor*)&input)->t->stride[1];
			break;

		case Tensor::Float:
			m_input_stride_h = ((const FloatTensor*)&input)->t->stride[0];
			m_input_stride_w = ((const FloatTensor*)&input)->t->stride[1];
			break;

		case Tensor::Double:
			m_input_stride_h = ((const DoubleTensor*)&input)->t->stride[0];
			m_input_stride_w = ((const DoubleTensor*)&input)->t->stride[1];
			break;
		}

		updateIntegralFactors();
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
