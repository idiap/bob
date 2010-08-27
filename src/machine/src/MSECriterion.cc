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

	m_target->copy(target);

	double error_ = 0.0;
	for(int i = 0; i < m_target_size; i++)
	{
		double z = (*machine_output)(i) - (*m_target)(i);
    (*m_beta)(i) = 2.*z;
		error_ += z*z;
	}

  (*m_error)(0) = error_;
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
