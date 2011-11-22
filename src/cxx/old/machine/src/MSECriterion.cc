/**
 * @file cxx/old/machine/src/MSECriterion.cc
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
#include "machine/MSECriterion.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

MSECriterion::MSECriterion(const int target_size) : Criterion(target_size, 1)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

MSECriterion::~MSECriterion()
{
}


bool MSECriterion::forward(const DoubleTensor *machine_output, const Tensor *target)
{
	// Accept only 1D tensors

	if (machine_output->nDimension() != 1)
	{
		warning("MSECriterion::forward() : incorrect number of dimensions in machine output.");
		
		return false;
	}
	if (machine_output->size(0) != m_target_size)
	{
		warning("MSECriterion::forward() : incorrect input size along dimension 0 in machine output.");
		
		return false;
	}

	if (target->nDimension() != 1)
	{
		warning("MSECriterion::forward() : incorrect number of dimensions in target.");
		
		return false;
	}
	if (target->size(0) != m_target_size)
	{
		warning("MSECriterion::forward() : incorrect input size along dimension 0 in target.");
		
		return false;
	}

	((Tensor*)m_target)->copy(target);

	double error_ = 0.0;
	for(int i = 0; i < m_target_size; i++)
	{
		const double z = (*machine_output)(i) - (*m_target)(i);
    (*m_beta)(i) = 2.*z;
		error_ += z*z;
	}

  (*m_error)(0) = error_;
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
