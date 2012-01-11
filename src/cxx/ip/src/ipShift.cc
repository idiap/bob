/**
 * @file cxx/ip/src/ipShift.cc
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
#include "ip/ipShift.h"
#include "core/Tensor.h"

namespace bob {

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
	// Accept only 3D tensors of bob::Image type
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
  t_output->fill(0);

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	const int width = input.size(1);
	const int height = input.size(0);
	const int n_planes = input.size(2);
	const int start_x = getInRange(dx, 0, width);
	const int start_y = getInRange(dy, 0, height);
	const int stop_x = getInRange(width + dx, 0, width);
	const int stop_y = getInRange(height + dy, 0, height);
  for (int x=start_x; x<stop_x; ++x)
    for (int y=start_y; y<stop_y; ++y)
      for (int p=0; p<n_planes; ++p)
        (*t_output)(y, x, p) = (*t_input)(y-start_y, x-start_x, p);

	return true;
}

/////////////////////////////////////////////////////////////////////////

}
