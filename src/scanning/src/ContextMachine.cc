#include "scanning/ContextMachine.h"
#include "core/File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ContextMachine::ContextMachine()
	:	m_foutputs(NoFeatures),
		m_fmodels(new LRMachine[NoFeatures])
{
	m_output.resize(1);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ContextMachine::~ContextMachine()
{
	delete[] m_fmodels;
}

/////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool ContextMachine::forward(const Tensor& input)
{
	// Extract features from input tensor
	if (	input.getDatatype() != Tensor::Double ||
		m_context.copyFrom((const DoubleTensor&)input) == false)
	{
		Torch::message("ContextMachine::forward - invalid input!\n");
		return false;
	}

	// Pass the features to the feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].forward(m_context.m_features[f]) == false)
		{
			Torch::message("ContextMachine::forward - failed to run some feature model!\n");
			return false;
		}

		const double score = ((const DoubleTensor&)m_fmodels[f].getOutput()).get(0);
		m_foutputs.set(f, score - m_fmodels[f].getThreshold());
	}

	// Final decision: run the combined classifier
	m_patternClass = 0;

	if (m_cmodel.forward(m_foutputs) == false)
	{
		Torch::message("ContextMachine::forward - failed to run the combined classifier!\n");
		return false;
	}

	const double score = ((const DoubleTensor&)m_cmodel.getOutput()).get(0);
  m_output(0) = score;
	m_isPattern = score >= m_cmodel.getThreshold();

	// OK
	m_confidence = m_output.get(0);
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options})

bool ContextMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, 1, "ID") != 1)
	{
		Torch::message("ContextMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("ContextMachine::load - invalid <ID>, this is not a ContextMachine model!\n");
		return false;
	}

	// Load the context feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].loadFile(file) == false)
		{
			message("ContextMachine::loadFile - invalid feature model [%d/%d]!\n",
				f + 1, NoFeatures);
			return false;
		}
	}

	// Load the combined classifier
	if (m_cmodel.loadFile(file) == false)
	{
		message("ContextMachine::loadFile - invalid combined model!\n");
		return false;
	}

	// OK
	return true;
}

bool ContextMachine::saveFile(File& file) const
{
	// Write the ID
	const int id = getID();
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("ContextMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the context feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].saveFile(file) == false)
		{
			Torch::message("ContextMachine::save - failed to write the feature model [%d/%d]!\n",
				f + 1, NoFeatures);
			return false;
		}
	}

	// Write the combined classifier
	if (m_cmodel.saveFile(file) == false)
	{
		Torch::message("ContextMachine::save - failed to write the combined model!\n");
		return true;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
