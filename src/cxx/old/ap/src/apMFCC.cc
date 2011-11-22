/**
 * @file cxx/old/ap/src/apMFCC.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
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
#include "ap/apMFCC.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

apMFCC::apMFCC()
	:	apCore()
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

apMFCC::~apMFCC()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool apMFCC::checkInput(const Tensor& input) const
{
	// Accept only tensors of Torch::Float
	if (input.getDatatype() != Tensor::Float) return false;


	if (input.nDimension() != 1)
	{
		warning("apMFCC(): input dimension should be 1.");
		return false;
	}
	
	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool apMFCC::allocateOutput(const Tensor& input)
{
	if (	m_output == 0 )
	{
		cleanup();
	
		if (input.nDimension() == 1)
		{
			print("apMFCC::allocateOutput() MFCC ...\n");

			int N = input.size(0);

			m_n_outputs = 1;
			m_output = new Tensor*[m_n_outputs];
			m_output[0] = new FloatTensor(N);
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool apMFCC::processInput(const Tensor& input)
{
	const FloatTensor* t_input = (FloatTensor*)&input;

	m_output[0]->copy(t_input);

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}

