#include "ipCore.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

ipCore::ipCore()
	: 	spCore(),
		m_inputSize(0, 0)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

ipCore::~ipCore()
{
}

//////////////////////////////////////////////////////////////////////////
// Change the input image size

bool ipCore::setInputSize(const sSize& new_size)
{
	if (new_size.w > 0 && new_size.h > 0)
	{
		if (	m_inputSize.w != new_size.w ||
			m_inputSize.h != new_size.h)
		{
			// Delete the old tensors (if any)
			cleanup();
			m_inputSize = new_size;
		}
		return true;
	}
	return false;
}

bool ipCore::setInputSize(int new_w, int new_h)
{
	if (new_w > 0 && new_h > 0)
	{
		if (	m_inputSize.w != new_w ||
			m_inputSize.h != new_h)
		{
			// Delete the old tensors (if any)
			cleanup();
			m_inputSize.w = new_w;
			m_inputSize.h = new_h;
		}
		return true;
	}
	return false;
}

//////////////////////////////////////////////////////////////////////////
// Retrieve the input image size

int ipCore::getInputWidth() const
{
	return m_inputSize.w;
}

int ipCore::getInputHeight() const
{
	return m_inputSize.h;
}

//////////////////////////////////////////////////////////////////////////
}

