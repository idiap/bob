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
  const int K=input->sizeAll();
  for (int k=0; k<K; ++k) m_output(k) = exp((*input)(k));
	return true;
}

bool Exp::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
	for(int i = 0; i < n_inputs; i++) (*m_beta)(i) = (*alpha)(i) * m_output(i);
	return true;
}


//////////////////////////////////////////////////////////////////////////
}
