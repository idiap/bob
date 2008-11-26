#include "ipSWDummyPruner.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ipSWDummyPruner::ipSWDummyPruner(Type type)
	:	ipSWPruner(),
		m_type(type)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipSWDummyPruner::~ipSWDummyPruner()
{
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipSWDummyPruner::checkInput(const Tensor& input) const
{
	// TODO
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipSWDummyPruner::allocateOutput(const Tensor& input)
{
	// TODO
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipSWDummyPruner::processInput(const Tensor& input)
{
	// Check if the sub-window should be rejected or not
	switch (m_type)
	{
	case RejectAll:
		m_isRejected = true;
		break;

	case RejectNone:
		m_isRejected = false;
		break;

	case RejectRandom:
		m_isRejected = rand() % 8 != 0;
		break;
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////

}
