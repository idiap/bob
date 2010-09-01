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
