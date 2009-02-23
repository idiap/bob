#include "spCore.h"
#include "general.h"
#include "Tensor.h"

namespace Torch {

////////////////////////////////////////////////////////////////////
// Constructor

spCore::spCore()
	:	Object(),
		m_output(0),
		m_n_outputs(0)
{
}

////////////////////////////////////////////////////////////////////
// Destructor

spCore::~spCore()
{
	cleanup();
}

////////////////////////////////////////////////////////////////////
// Deallocate allocated output tensors

void spCore::cleanup()
{
	for (int i = 0; i < m_n_outputs; i ++)
	{
		delete m_output[i];
	}
	delete[] m_output;

	m_output = 0;
	m_n_outputs = 0;
}

////////////////////////////////////////////////////////////////////
// Process some input tensor

bool spCore::process(const Tensor& input)
{
	// Check if the input tensor has the right dimensions and type
	if (checkInput(input) == false)
	{
		Torch::message("Torch::spCore::process - the input tensor is invalid!\n");
		return false;
	}

	// Allocate (if needed) the output tensor given the input tensor dimensions
	if (allocateOutput(input) == false)
	{
		Torch::message("Torch::spCore::process - cannot allocate output tensors!\n");
		return false;
	}

	// OK, now do the processing ...
	return processInput(input);
}

////////////////////////////////////////////////////////////////////
// Access the results

int spCore::getNOutputs() const
{
	return m_n_outputs;
}

const Tensor& spCore::getOutput(int index) const
{
	if (index < 0 || index >= m_n_outputs)
	{
		Torch::error("Torch::spCore::getOutput - invalid index!");
	}
	return *m_output[index];
}

////////////////////////////////////////////////////////////////////

bool spCore::saveFile(File& file) const
{
	return true;
}
		
}


