#include "machine/Criterion.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

Criterion::Criterion(const int target_size, const int error_size)
{
   	m_target_size = target_size;
   	m_error_size = error_size;
	m_error = NULL;
	m_beta = NULL;
	m_error = new DoubleTensor(m_error_size);
	m_beta = new DoubleTensor(m_target_size);
	m_target = new DoubleTensor(m_target_size);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

Criterion::~Criterion()
{
        // Cleanup
	delete m_error;
	delete m_beta;
	delete m_target;
}

//////////////////////////////////////////////////////////////////////////
}
