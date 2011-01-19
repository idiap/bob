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
  const int K=input->sizeAll();
  for (int k=0; k<K; ++k) m_output(k) = log((*input)(k));
	return true;
}

bool Log::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
   	warning("Log::backward() not implemented.");

	return false;
}

//////////////////////////////////////////////////////////////////////////
}
