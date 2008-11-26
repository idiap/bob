#include "ipSWDummyEvaluator.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ipSWDummyEvaluator::ipSWDummyEvaluator(int modelWidth, int modelHeight, Type type)
	: 	ipSWEvaluator(),
		m_modelWidth(modelWidth), m_modelHeight(modelHeight),
		m_modelThreshold(0.0f),
		m_type(type)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipSWDummyEvaluator::~ipSWDummyEvaluator()
{
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipSWDummyEvaluator::checkInput(const Tensor& input) const
{
	// TODO
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipSWDummyEvaluator::allocateOutput(const Tensor& input)
{
	// TODO
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipSWDummyEvaluator::processInput(const Tensor& input)
{
	// Check if pattern ...
	switch (m_type)
	{
	case PassAll:
		m_confidence = 1.0f;
		break;

	case PassNone:
		m_confidence = -1.0f;
		break;

	case PassRandom:
		m_confidence = (rand() % 1024 - 1000.0f) / 1024.0f;
		break;
	}
	m_isPattern = m_confidence >= m_modelThreshold;

	return true;
}

/////////////////////////////////////////////////////////////////////////

}
