#include "machine/Sigmoid.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Sigmoid::Sigmoid() : GradientMachine()
{
}

Sigmoid::Sigmoid(const int n_units_) : GradientMachine(n_units_, n_units_)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Sigmoid::~Sigmoid()
{
}


//////////////////////////////////////////////////////////////////////////

bool Sigmoid::forward(const DoubleTensor *input)
{
  const int K=input->sizeAll();
  for (int k=0; k<K; ++k) m_output(k) = (1./(1. + exp(-(*input)(k))));
	return true;
}

bool Sigmoid::backward(const DoubleTensor *input, const DoubleTensor *alpha)
{
  //int n_outputs = m_parameters->getI("n_outputs");
	for(int i = 0; i < n_outputs; i++)
	{
		double z = m_output(i);
		(*m_beta)(i) = (*alpha)(i) * (1. - z) * z;
	}

	return true;
}


//////////////////////////////////////////////////////////////////////////
}
