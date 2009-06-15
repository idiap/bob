#include "ProfileMachine.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ProfileMachine::ProfileMachine()
	:	m_foutputs(NoFeatures),
		m_fmodels(new LRMachine[NoFeatures])
{
	m_output.resize(1);
	m_poutput = (double*)m_output.dataW();
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ProfileMachine::~ProfileMachine()
{
	delete[] m_fmodels;
}

/////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool ProfileMachine::forward(const Tensor& input)
{
	// Extract features from input tensor
	if (	input.getDatatype() != Tensor::Double ||
		m_profile.copyFrom((const DoubleTensor&)input) == false)
	{
		Torch::message("ProfileMachine::forward - invalid input!\n");
		return false;
	}

	// Pass the features to the feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].forward(m_profile.m_features[f]) == false)
		{
			Torch::message("ProfileMachine::forward - failed to run some feature model!\n");
			return false;
		}

		const double score = ((const DoubleTensor&)m_fmodels[f].getOutput()).get(0);
		m_foutputs.set(f, score >= m_fmodels[f].getThreshold() ? 1.0 : -1.0);
	}

	// Final decision: run the combined classifier
	m_patternClass = 0;

	if (m_cmodel.forward(m_foutputs) == false)
	{
		Torch::message("ProfileMachine::forward - failed to run the combined classifier!\n");
		return false;
	}

	const double score = ((const DoubleTensor&)m_cmodel.getOutput()).get(0);
	*m_poutput = score;
	m_isPattern = score >= m_cmodel.getThreshold();

	// OK
	m_confidence = *m_poutput;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options})

bool ProfileMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("ProfileMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("ProfileMachine::load - invalid <ID>, this is not a ProfileMachine model!\n");
		return false;
	}

	// Load the profile feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].loadFile(file) == false)
		{
			message("ProfileMachine::loadFile - invalid feature model [%d/%d]!\n",
				f + 1, NoFeatures);
			return false;
		}
	}

	// Load the combined classifier
	if (m_cmodel.loadFile(file) == false)
	{
		message("ProfileMachine::loadFile - invalid combined model!\n");
		return false;
	}

	// OK
	return true;
}

bool ProfileMachine::saveFile(File& file) const
{
	// Write the ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("ProfileMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the profile feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].saveFile(file) == false)
		{
			Torch::message("ProfileMachine::save - failed to write the feature model [%d/%d]!\n",
				f + 1, NoFeatures);
			return false;
		}
	}

	// Write the combined classifier
	if (m_cmodel.saveFile(file) == false)
	{
		Torch::message("ProfileMachine::save - failed to write the combined model!\n");
		return true;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
