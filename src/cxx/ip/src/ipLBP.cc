/**
 * @file cxx/ip/src/ipLBP.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "ip/ipLBP.h"
#include "core/Tensor.h"

namespace bob {

/////////////////////////////////////////////////////////////////////////
// Resize to a new model size

void ipLBP::IntegralFactors::resizeModel(int model_w, int model_h)
{
	m_model_w = model_w;
	m_model_h = model_h;
}

/////////////////////////////////////////////////////////////////////////
// Resize to a new subwindow size

void ipLBP::IntegralFactors::resizeSW(int sw_w, int sw_h, int stride_w, int stride_h, int mask_x, int mask_y, int mask_radius)
{
	if (stride_w == 0 || stride_h == 0)
	{
		return;
	}

	m_sw_w = sw_w;
	m_sw_h = sw_h;

	// Scalling factors
	const double scale_w = (m_sw_w + 0.0) / (m_model_w + 0.0);
	const double scale_h = (m_sw_h + 0.0) / (m_model_h + 0.0);
	const int mask_size = 2 * mask_radius + 1;

	// Compute the displacement from the subwindow top-left corner
	m_dx = stride_w * (FixI(mask_x * scale_w) - 1);
	m_dy = stride_h * (FixI(mask_y * scale_h) - 1);

	// Compute the cell width
	m_cell_w = stride_w * FixI(mask_size * scale_w);
	m_cell_w1 = FixI((double)m_cell_w / 3);
	m_cell_w12 = m_cell_w - m_cell_w1;

	// Compute the cell height
	m_cell_h = stride_h * FixI(mask_size * scale_h);
	m_cell_h1 = FixI((double)m_cell_h / 3);
	m_cell_h12 = m_cell_h - m_cell_h1;
}

/////////////////////////////////////////////////////////////////////////
// Constructor

ipLBP::ipLBP(int P, int R)
	:	ipCore(),
		m_P(P), m_R(R),
		m_x(0), m_y(0),
		m_input_w(0), m_input_h(0), m_input_stride_w(0), m_input_stride_h(0),
		m_lbp(0),
		m_lut_RI(0),
		m_lut_U2(0),
		m_lut_U2RI(0),
		m_lut_addAvgBit(0),
		m_lut_normal(0),
		m_crt_lut(0),
		m_toAverage(false), m_addAvgBit(false), m_uniform(0), m_rot_invariant(false),
		m_need_interp(false)
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
/// Change the region of the input tensor to process - overriden

void ipLBP::setRegion(const TensorRegion& region)
{
	const bool changed = 	m_region.size[0] != region.size[0] ||
				m_region.size[1] != region.size[1];

	ipCore::setRegion(region);
	if (changed == true)
	{
		m_ii_factors.resizeSW(m_region.size[1], m_region.size[0], m_input_stride_w, m_input_stride_h, m_x, m_y, m_R);
		m_need_interp = (m_modelSize.size[0] != m_region.size[0]) || (m_modelSize.size[1] != m_region.size[1]);
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/// Change the model size (if used with some machine) - overriden

void ipLBP::setModelSize(const TensorSize& modelSize)
{
	const bool changed = 	m_modelSize.size[0] != modelSize.size[0] ||
				m_modelSize.size[1] != modelSize.size[1];

	ipCore::setModelSize(modelSize);
	if (changed == true)
	{
		m_ii_factors.resizeModel(m_modelSize.size[1], m_modelSize.size[0]);
	}
	m_need_interp = (m_modelSize.size[0] != m_region.size[0]) || (m_modelSize.size[1] != m_region.size[1]);
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
                IntTensor* t_output = (IntTensor*)m_output[0];
                m_lbp = &(*t_output)(0);
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
			m_input_stride_h = ((const CharTensor*)&input)->stride(0);
			m_input_stride_w = ((const CharTensor*)&input)->stride(1);
			break;

		case Tensor::Short:
			m_input_stride_h = ((const ShortTensor*)&input)->stride(0);
			m_input_stride_w = ((const ShortTensor*)&input)->stride(1);
			break;

		case Tensor::Int:
			m_input_stride_h = ((const IntTensor*)&input)->stride(0);
			m_input_stride_w = ((const IntTensor*)&input)->stride(1);
			break;

		case Tensor::Long:
			m_input_stride_h = ((const LongTensor*)&input)->stride(0);
			m_input_stride_w = ((const LongTensor*)&input)->stride(1);
			break;

		case Tensor::Float:
			m_input_stride_h = ((const FloatTensor*)&input)->stride(0);
			m_input_stride_w = ((const FloatTensor*)&input)->stride(1);
			break;

		case Tensor::Double:
			m_input_stride_h = ((const DoubleTensor*)&input)->stride(0);
			m_input_stride_w = ((const DoubleTensor*)&input)->stride(1);
			break;

    default:
      break;
		}

		m_ii_factors.resizeSW(m_region.size[1], m_region.size[0], m_input_stride_w, m_input_stride_h, m_x, m_y, m_R);
		m_need_interp = (m_modelSize.size[0] != m_region.size[0]) || (m_modelSize.size[1] != m_region.size[1]);
	}
	return true;
}

}
