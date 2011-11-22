/**
 * @file cxx/ip/src/ipRescaleGray.cc
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
#include "ip/ipRescaleGray.h"

namespace Torch {

template <typename T> void computeRescaleGray(const Tensor& input, ShortTensor& t_output)
{
  const T& t_input = dynamic_cast<const T&>(input);
  const int height = t_input.size(0);
  const int width = t_input.size(1);
  const int n_planes = t_input.size(2);

  /* Start to "normalize" current values in range [0,255] */
	double max_val = t_input(0,0,0); /*STL std::numeric_limits<double>::min( ); */
	double min_val = t_input(0,0,0); /*STL std::numeric_limits<double>::max( ); */
	double range;

	/* find min and max values in the image */
	for( int p=0; p<n_planes; ++p )
	{
		for( int y=0; y<height; ++y )
		{
			for( int x=0; x<width; ++x )
			{
				if (t_input(y,x,p) > max_val)
					max_val = t_input(y,x,p);

				if (t_input(y,x,p) < min_val)
					min_val = t_input(y,x,p);
			}
		}
	}

	/* Compute the range */
	range = max_val - min_val;
	const double EPSILON = 1e-12; /*STL std::numeric_limits<double>::epsilon() * 1000; */
	bool range_zero = range < EPSILON;

	/* Change the scale */
	for( int p=0; p<n_planes; ++p )
	{
		for( int y=0; y<height; ++y )
		{
			for( int x=0; x<width; ++x )
			{
				t_output(y,x,p) = ( range_zero ? 0 : FixI(255. * (t_input(y,x,p) - min_val) / range) );
			}
		}
	}

}


/////////////////////////////////////////////////////////////////////////
// Constructor
ipRescaleGray::ipRescaleGray()
	:	ipCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor
ipRescaleGray::~ipRescaleGray()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipRescaleGray::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors 
	if (	input.nDimension() != 3 )
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions
bool ipRescaleGray::allocateOutput(const Tensor& input)
{
	// Allocate output if required
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
		m_output[0] = new ShortTensor( input.size(0), input.size(1), input.size(2) );
		return true;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)
bool ipRescaleGray::processInput(const Tensor& input)
{
	// Prepare direct access to output data
	ShortTensor* t_output = dynamic_cast<ShortTensor*>(m_output[0]);

	switch (input.getDatatype())
	{
		case Tensor::Char:
      computeRescaleGray<CharTensor>( input, *t_output);
			break;

		case Tensor::Short:
      computeRescaleGray<ShortTensor>( input, *t_output);
			break;

		case Tensor::Int:
      computeRescaleGray<IntTensor>( input, *t_output);
			break;

		case Tensor::Long:
      computeRescaleGray<LongTensor>( input, *t_output);
			break;

		case Tensor::Float:
      computeRescaleGray<FloatTensor>( input, *t_output);
			break;

		case Tensor::Double:
      computeRescaleGray<DoubleTensor>( input, *t_output);
			break;

    default:
      return false;
	}

	// OK
	return true;
}
/////////////////////////////////////////////////////////////////////////

}

