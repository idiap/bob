#include "machine/Log.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Log::Log() : GradientMachine()
{
}

Log::Log(const int n_units_) : GradientMachine(n_units_, n_units_)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Log::~Log()
{
}


//////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool Log::forward(const DoubleTensor *input)
{
	THTensor *src = input->t;
	THTensor *dst = m_output.t;
	
	TH_TENSOR_APPLY2(double, dst, double, src, *dst_p = log(*src_p););

	return true;
}

bool Log::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
   	warning("Log::backward() not implemented.");

	return false;
}

//////////////////////////////////////////////////////////////////////////
}
