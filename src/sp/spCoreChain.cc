#include "sp/spCoreChain.h"

namespace Torch {

////////////////////////////////////////////////////////////////////
// Constructor

spCoreChain::spCoreChain()
	:	spCore(),
		m_cores(0),
		m_n_cores(0)
{
}

////////////////////////////////////////////////////////////////////
// Destructor

spCoreChain::~spCoreChain()
{
	delete[] m_cores;
}

////////////////////////////////////////////////////////////////////
// Manage the chain of <spCore> to use

void spCoreChain::clear()
{
	delete[] m_cores;
	m_cores = 0;
	m_n_cores = 0;
}

bool spCoreChain::add(spCore* core)
{
	// Check parameters
	if (core == 0)
	{
		return false;
	}

	// Allocate a +1 cores
	spCore** new_cores = new spCore*[m_n_cores + 1];
	for (int i = 0; i < m_n_cores; i ++)
	{
		new_cores[i] = m_cores[i];
	}
	delete[] m_cores;
	m_cores = new_cores;

	// OK
	m_cores[m_n_cores ++] = core;
	return true;
}

////////////////////////////////////////////////////////////////////
/// Access the results

int spCoreChain::getNOutputs() const
{
	if (m_n_cores < 1)
	{
		error("spCoreChain::getNOutputs - no spCore set!\n");
	}

	return m_cores[m_n_cores - 1]->getNOutputs();
}

const Tensor& spCoreChain::getOutput(int index) const
{
	if (m_n_cores < 1)
	{
		error("spCoreChain::getOutput - no spCore set!\n");
	}

	return m_cores[m_n_cores - 1]->getOutput(index);
}

////////////////////////////////////////////////////////////////////
/// Check if the input tensor has the right dimensions and type

bool spCoreChain::checkInput(const Tensor& input) const
{
	return true;
}

////////////////////////////////////////////////////////////////////
/// Allocate (if needed) the output tensors given the input tensor dimensions

bool spCoreChain::allocateOutput(const Tensor& input)
{
	return true;
}

////////////////////////////////////////////////////////////////////
/// Process some input tensor (the input is checked, the outputs are allocated)

bool spCoreChain::processInput(const Tensor& input)
{
	if (m_n_cores < 1)
	{
		error("spCoreChain::processInput - no spCore set!\n");
	}

	if (m_cores[0]->process(input) == false)
	{
		print("spCoreChain::process - failed to run process!\n");
		return false;
	}

	for (int i = 0; i < m_n_cores - 1; i ++)
	{
		if (m_cores[i]->getNOutputs() != 1)
		{
			print("spCoreChain::process - expected one single output!\n");
			return false;
		}

		if (m_cores[i + 1]->process(m_cores[i]->getOutput(0)) == false)
		{
			print("spCoreChain::process - failed to run process!\n");
			return false;
		}
	}

	return true;
}

////////////////////////////////////////////////////////////////////
// Change the region of the input tensor to process

void spCoreChain::setRegion(const TensorRegion& region)
{
	for (int i = 0; i < m_n_cores; i ++)
	{
		m_cores[i]->setRegion(region);
	}
}

////////////////////////////////////////////////////////////////////
// Change the model size (if used with some machine)

void spCoreChain::setModelSize(const TensorSize& modelSize)
{
	for (int i = 0; i < m_n_cores; i ++)
	{
		m_cores[i]->setModelSize(modelSize);
	}
}

////////////////////////////////////////////////////////////////////

}


