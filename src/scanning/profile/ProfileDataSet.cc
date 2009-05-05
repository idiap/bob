#include "ProfileDataSet.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ProfileDataSet::ProfileDataSet(int pf_feature)
	: 	m_feature(pf_feature),
		m_profiles(0),
		m_masks(0),
		m_capacity(0),
		m_target_neg(1),
		m_target_pos(1)
{
	m_target_neg.fill(-1.0);
	m_target_pos.fill(1.0);

	reset(pf_feature);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ProfileDataSet::~ProfileDataSet()
{
	for (int i = 0; i < m_capacity; i ++)
	{
		delete m_profiles[i];
	}
	delete[] m_profiles;
	delete[] m_masks;
}

/////////////////////////////////////////////////////////////////////////
// Reset to new profile feature

void ProfileDataSet::reset(int pf_feature)
{
	m_feature = pf_feature;
}

/////////////////////////////////////////////////////////////////////////
// Access examples - overriden

Tensor* ProfileDataSet::getExample(long index)
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ProfileDataSet::getExample - invalid index!\n");
	}
	return &m_profiles[index]->m_features[m_feature];
}

Tensor& ProfileDataSet::operator()(long index)
{
	return *getExample(index);
}

/////////////////////////////////////////////////////////////////////////
// Access targets - overriden

Tensor* ProfileDataSet::getTarget(long index)
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ProfileDataSet::getTarget - invalid index!\n");
	}

	const unsigned char mask = m_masks[index];
	return mask == Negative ? &m_target_neg : &m_target_pos;
}

void ProfileDataSet::setTarget(long index, Tensor* target)
{
	// Nothing to do here!
}

/////////////////////////////////////////////////////////////////////////
// Profile access in the distribution

const Profile* ProfileDataSet::getProfile(long index) const
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ProfileDataSet::getProfile - invalid index!\n");
	}

	return m_profiles[index];
}

bool ProfileDataSet::isPosProfile(long index) const
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ProfileDataSet::isPosProfile - invalid index!\n");
	}

	return m_masks[index] != Negative;
}

/////////////////////////////////////////////////////////////////////////
// Clear cumulated profiles

void ProfileDataSet::clear()
{
	m_n_examples = 0;
}

/////////////////////////////////////////////////////////////////////////
// Cumulate a new profile (negative, positive or ground truth)

void ProfileDataSet::cumulate(bool positive, const Profile& profile)
{
	static const int increment = 1024;

	if (m_n_examples >= m_capacity)
	{
		m_profiles = resize(m_profiles, m_capacity, increment);
		m_masks = resize(m_masks, m_capacity, increment);
		m_capacity += increment;
	}

	*m_profiles[m_n_examples] = profile;
	m_masks[m_n_examples] = positive == true ? Positive : Negative;
	m_n_examples ++;
}

void ProfileDataSet::cumulate(const Profile& gt_profile)
{
	static const int increment = 1024;

	// Add the detection
	if (m_n_examples >= m_capacity)
	{
		m_profiles = resize(m_profiles, m_capacity, increment);
		m_masks = resize(m_masks, m_capacity, increment);
		m_capacity += increment;
	}

	*m_profiles[m_n_examples] = gt_profile;
	m_masks[m_n_examples] = GroundTruth;
	m_n_examples ++;
}

/////////////////////////////////////////////////////////////////////////
// Resize some distribution to fit new samples

Profile** ProfileDataSet::resize(Profile** old_data, int capacity, int increment)
{
	Profile** new_data = new Profile*[capacity + increment];
	for (int i = 0; i < capacity; i ++)
	{
		new_data[i] = old_data[i];
	}
	for (int i = 0; i < increment; i ++)
	{
		new_data[capacity + i] = new Profile;
	}
	delete[] old_data;

	return new_data;
}

unsigned char* ProfileDataSet::resize(unsigned char* old_data, int capacity, int increment)
{
	unsigned char* new_data = new unsigned char[capacity + increment];
	for (int i = 0; i < capacity; i ++)
	{
		new_data[i] = old_data[i];
	}
	for (int i = 0; i < increment; i ++)
	{
		new_data[capacity + i] = 0x00;
	}
	delete[] old_data;

	return new_data;
}

/////////////////////////////////////////////////////////////////////////
// Save the distribution

void ProfileDataSet::save(const char* dir_data, const char* name) const
{
	char str[512];

	// Negative samples
	sprintf(str, "%s_%s_neg", dir_data, name);
	save(str, Negative);

	// Positive samples
	sprintf(str, "%s_%s_pos", dir_data, name);
	save(str, Positive);

	// Ground truth samples
	sprintf(str, "%s_%s_gt", dir_data, name);
	save(str, GroundTruth);
}

void ProfileDataSet::save(const char* basename, unsigned char mask) const
{
	// Open the files
	for (int f = 0; f < NoFeatures; f ++)
	{
		char str[512];
		sprintf(str, "%s_%s.data", basename, FeatureNames[f]);

		File file;
		CHECK_FATAL(file.open(str, "w+") == true);

		const int fsize = FeatureSizes[f];
		for (int i = 0; i < m_n_examples; i ++)
			if (m_masks[i] == mask)
			{
				const double* values = (const double*)m_profiles[i]->m_features[f].dataR();
				for (int k = 0; k < fsize; k ++)
				{
					file.printf("%f\t", values[k]);
				}
				file.printf("\n");
			}

		file.close();
	}
}

/////////////////////////////////////////////////////////////////////////

}
