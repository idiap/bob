/**
 * @file cxx/old/machine/src/TwoClassNLLCriterion.cc
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
#include "machine/TwoClassNLLCriterion.h"
#include "machine/Machine.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

TwoClassNLLCriterion::TwoClassNLLCriterion(const double cst) : Criterion(1, 1)
{
	m_cst = cst;
}

//////////////////////////////////////////////////////////////////////////
// Destructor

TwoClassNLLCriterion::~TwoClassNLLCriterion()
{
}


bool TwoClassNLLCriterion::forward(const DoubleTensor *machine_output, const Tensor *target)
{
	// Accept only 1D tensors

	if (machine_output->nDimension() != 1)
	{
		warning("TwoClassNLLCriterion::forward() : incorrect number of dimensions in machine output.");
		
		return false;
	}
	if (machine_output->size(0) != m_target_size)
	{
		warning("TwoClassNLLCriterion::forward() : incorrect input size along dimension 0 in machine output.");
		
		return false;
	}

	if (target->nDimension() != 1)
	{
		warning("TwoClassNLLCriterion::forward() : incorrect number of dimensions in target.");
		
		return false;
	}
	if (target->size(0) != m_target_size)
	{
		warning("TwoClassNLLCriterion::forward() : incorrect input size along dimension 0 in target.");
		
		return false;
	}

	((Tensor*)m_target)->copy(target);

	double error = Torch::log_add(0, m_cst - (*m_target)(0) * (*machine_output)(0));
	(*m_error)(0) = error;
	(*m_beta)(0) = - (*m_target)(0) * (1. - exp(-error));

	return true;
}

//////////////////////////////////////////////////////////////////////////
}
