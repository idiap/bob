#include "machine/Exp.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Exp::Exp() : GradientMachine()
{
}

Exp::Exp(const int n_units_) : GradientMachine(n_units_, n_units_)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Exp::~Exp()
{
}


//////////////////////////////////////////////////////////////////////////

bool Exp::forward(const DoubleTensor *input)
{
	THTensor *src = input->t;
	THTensor *dst = m_output.t;
	
	TH_TENSOR_APPLY2(double, dst, double, src, *dst_p = exp(*src_p););

	return true;
}

bool Exp::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
	double *beta_ = (double *) m_beta->dataW();
	double *alpha_ = (double *) alpha->dataR();
	double *output_ = (double *) m_output.dataR();

	//int n_inputs = m_parameters->getI("n_inputs");

	for(int i = 0; i < n_inputs; i++)
		beta_[i] = alpha_[i] * output_[i];

	return true;
}


//////////////////////////////////////////////////////////////////////////
}
