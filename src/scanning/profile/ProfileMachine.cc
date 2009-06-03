#include "ProfileMachine.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ProfileMachine::ProfileMachine()
	:	m_foutputs(NoFeatures),
		m_fmodels(new FLDAMachine[NoFeatures]),
		m_fselected(new unsigned char[NoFeatures])
{
	m_output.resize(1);
	m_poutput = (double*)m_output.dataW();

	for (int f = 0; f < NoFeatures; f ++)
	{
		m_fselected[f] = 0x01;
	}
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ProfileMachine::~ProfileMachine()
{
	delete[] m_fmodels;
	delete[] m_fselected;
}

/////////////////////////////////////////////////////////////////////////
// Change the selection state for some profile model

void ProfileMachine::setFSelected(int f, bool selected)
{
	m_fselected[f] = selected ? 0x01 : 0x00;

	// Resize the buffer
	int n_selected = 0;
	for (int f = 0; f < NoFeatures; f ++)
		if (isFSelected(f) == true)
	{
		n_selected ++;
	}
	m_foutputs.resize(n_selected);
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
	int index = 0;
	for (int f = 0; f < NoFeatures; f ++)
		if (isFSelected(f) == true)
		{
			if (m_fmodels[f].forward(m_profile.m_features[f]) == false)
			{
				Torch::message("ProfileMachine::forward - failed to run some feature model!\n");
				return false;
			}

			const double score = ((const DoubleTensor&)m_fmodels[f].getOutput()).get(0);
			m_foutputs.set(index, score >= m_fmodels[f].getThreshold() ? 1.0 : -1.0);
			index ++;
		}

	// Final decision: run the classifier
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

	// Load the selected profile features
	if (file.taggedRead(m_fselected, sizeof(unsigned char), NoFeatures, "SELECTED") != NoFeatures)
	{
		Torch::message("ProfileMachine::load - invalid <SELECTED> field!\n");
		return false;
	}
	int n_selected = 0;
	for (int f = 0; f < NoFeatures; f ++)
		if (isFSelected(f) == true)
	{
		n_selected ++;
	}
	m_foutputs.resize(n_selected);

	// Load the profile feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].loadFile(file) == false)
		{
			print("ProfileMachine::loadFile - step3.1\n");
			return false;
		}
	}

	// Load the combined classifier
	return m_cmodel.loadFile(file);
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

	// Write the selected profile features
	if (file.taggedWrite(m_fselected, sizeof(unsigned char), NoFeatures, "SELECTED") != NoFeatures)
	{
		Torch::message("ProfileMachine::save - failed to write <SELECTED> field!\n");
		return false;
	}

	// Write the profile feature models
	for (int f = 0; f < NoFeatures; f ++)
	{
		if (m_fmodels[f].saveFile(file) == false)
		{
			return false;
		}
	}

	// Write the combined classifier
	return m_cmodel.saveFile(file);
}

/////////////////////////////////////////////////////////////////////////

}
