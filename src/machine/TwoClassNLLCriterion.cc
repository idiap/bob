#include "machine/TwoClassNLLCriterion.h"

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

	m_target->copy(target);

	double *o_ = (double *) machine_output->dataR();
	double *t_ = (double *) m_target->dataR();
	double *beta_ = (double *) m_beta->dataW();
	double *e_ = (double *) m_error->dataW();

	double error = THLogAdd(0, m_cst - t_[0] * o_[0]);

	*e_ = error;
	beta_[0] = - t_[0] * (1. - exp(-error));

	return true;
}

//////////////////////////////////////////////////////////////////////////
}
