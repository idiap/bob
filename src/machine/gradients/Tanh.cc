#include "Tanh.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Tanh::Tanh() : GradientMachine()
{
}

Tanh::Tanh(const int n_units_) : GradientMachine(n_units_, n_units_)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Tanh::~Tanh()
{
}


//////////////////////////////////////////////////////////////////////////

bool Tanh::forward(const DoubleTensor *input)
{
	THTensor *src = input->t;
	THTensor *dst = m_output.t;
	
	TH_TENSOR_APPLY2(double, dst, double, src, *dst_p = tanh(*src_p););

	return true;
}

bool Tanh::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
   	//int n_outputs = m_parameters->getI("n_outputs");

	double *beta_ = (double *) m_beta->dataW();
	double *alpha_ = (double *) alpha->dataR();
	double *output_ = (double *) m_output.dataR();

	for(int i = 0; i < n_outputs; i++)
	{
		double z = output_[i];

		beta_[i] = alpha_[i] * (1. - z*z);
	}
	
	return true;
}


//////////////////////////////////////////////////////////////////////////
}
