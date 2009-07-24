#include "ProfileDataSet.h"
#include "File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ProfileDataSet::ProfileDataSet(int pf_feature)
	: 	DataSet(Tensor::Double, true, Tensor::Double),
		m_feature(pf_feature),
		m_profiles(0),
		m_masks(0),
		m_capacity(0),
		m_target_neg(1),
		m_target_pos(1)
{
	m_target_neg.fill(0.1);
	m_target_pos.fill(0.9);

	reset(pf_feature);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ProfileDataSet::~ProfileDataSet()
{
	cleanup();
}

/////////////////////////////////////////////////////////////////////////
// Delete stored profiles

void ProfileDataSet::cleanup()
{
	for (long i = 0; i < m_capacity; i ++)
	{
		delete m_profiles[i];
	}
	delete[] m_profiles;
	m_profiles = 0;

	delete[] m_masks;
	m_masks = 0;

	m_capacity = 0;
	m_n_examples = 0;
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

Profile** ProfileDataSet::resize(Profile** old_data, long capacity, long increment)
{
	Profile** new_data = new Profile*[capacity + increment];
	for (long i = 0; i < capacity; i ++)
	{
		new_data[i] = old_data[i];
	}
	for (long i = 0; i < increment; i ++)
	{
		new_data[capacity + i] = new Profile;
	}
	delete[] old_data;

	return new_data;
}

unsigned char* ProfileDataSet::resize(unsigned char* old_data, long capacity, long increment)
{
	unsigned char* new_data = new unsigned char[capacity + increment];
	for (long i = 0; i < capacity; i ++)
	{
		new_data[i] = old_data[i];
	}
	for (long i = 0; i < increment; i ++)
	{
		new_data[capacity + i] = 0x00;
	}
	delete[] old_data;

	return new_data;
}

/////////////////////////////////////////////////////////////////////////
// Save the distribution

bool ProfileDataSet::save(const char* dir_data, const char* name) const
{
	char str[1024];

	// Save separate profiles for plotting
	{
		// Negative samples
		sprintf(str, "%s_%s_neg", dir_data, name);
		if (save(str, Negative) == false)
		{
			return false;
		}

		// Positive samples
		sprintf(str, "%s_%s_pos", dir_data, name);
		if (save(str, Positive) == false)
		{
			return false;
		}

		// Ground truth samples
		sprintf(str, "%s_%s_gt", dir_data, name);
		if (save(str, GroundTruth) == false)
		{
			return false;
		}
	}

	// Save altogether in a binary format
	{
		// Open the file
		File file;
		sprintf(str, "%s_%s.distribution", dir_data, name);
		if (file.open(str, "w") == false)
		{
			return false;
		}

		// Write the number of samples
		if (file.taggedWrite(&m_n_examples, sizeof(long), 1, "NO") != 1)
		{
			print("ProfileDataSet::save - failed to write <NO> tag!\n");
			return false;
		}

		// Write the masks (positive, negative, ground truth)
		if (file.taggedWrite(m_masks, sizeof(unsigned char), m_n_examples, "MASKS") != m_n_examples)
		{
			print("ProfileDataSet::save - failed to write <MASKS> tag!\n");
			return false;
		}

		// Write each profile
		for (long s = 0; s < m_n_examples; s ++)
		{
			const Profile* profile = m_profiles[s];

			if (	file.write(&profile->m_pattern.m_x, sizeof(short), 1) != 1 ||
				file.write(&profile->m_pattern.m_y, sizeof(short), 1) != 1 ||
				file.write(&profile->m_pattern.m_w, sizeof(short), 1) != 1 ||
				file.write(&profile->m_pattern.m_h, sizeof(short), 1) != 1 ||
				file.write(&profile->m_pattern.m_confidence, sizeof(double), 1) != 1 ||
				file.write(&profile->m_pattern.m_activation, sizeof(short), 1) != 1)
			{
				print("ProfileDataSet::save - failed to write profile [%d/%d]!\n", s + 1, m_n_examples);
				return false;
			}

			for (int f = 0; f < NoFeatures; f ++)
			{
				if (file.write(	profile->m_features[f].dataR(), sizeof(double), FeatureSizes[f])
						!= FeatureSizes[f])
				{
					print("ProfileDataSet::save - failed to write profile [%d/%d]!\n", s + 1, m_n_examples);
					return false;
				}
			}
		}

		file.close();
	}

	// OK
	return true;
}

bool ProfileDataSet::save(const char* basename, unsigned char mask) const
{
	// Open the files
	for (int f = 0; f < NoFeatures; f ++)
	{
		char str[1024];
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

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Load the distribution

bool ProfileDataSet::load(const char* dir_data, const char* name)
{
	cleanup();

	// Open the file
	char str[1024];
	File file;
	sprintf(str, "%s_%s.distribution", dir_data, name);
	if (file.open(str, "r") == false)
	{
		return false;
	}

	// Read the number of samples
	if (file.taggedRead(&m_n_examples, sizeof(long), 1, "NO") != 1)
	{
		print("ProfileDataSet::save - failed to read <NO> tag!\n");
		return false;
	}

	// Allocate data
	m_masks = new unsigned char[m_n_examples];
	m_profiles = new Profile*[m_n_examples];
	for (long s = 0; s < m_n_examples; s ++)
	{
		m_profiles[s] = new Profile;
	}
	m_capacity = m_n_examples;

	// Read the masks (positive, negative, ground truth
	if (file.taggedRead(m_masks, sizeof(unsigned char), m_n_examples, "MASKS") != m_n_examples)
	{
		print("ProfileDataSet::save - failed to read <MASKS> tag!\n");
		return false;
	}

	// Read each profile
	for (long s = 0; s < m_n_examples; s ++)
	{
		Profile* profile = m_profiles[s];

		if (	file.read(&profile->m_pattern.m_x, sizeof(short), 1) != 1 ||
			file.read(&profile->m_pattern.m_y, sizeof(short), 1) != 1 ||
			file.read(&profile->m_pattern.m_w, sizeof(short), 1) != 1 ||
			file.read(&profile->m_pattern.m_h, sizeof(short), 1) != 1 ||
			file.read(&profile->m_pattern.m_confidence, sizeof(double), 1) != 1 ||
			file.read(&profile->m_pattern.m_activation, sizeof(short), 1) != 1)
		{
			print("ProfileDataSet::save - failed to read profile [%d/%d]!\n", s + 1, m_n_examples);
			return false;
		}

		for (int f = 0; f < NoFeatures; f ++)
		{
			if (file.read(	profile->m_features[f].dataW(),
					sizeof(double),
					FeatureSizes[f]) != FeatureSizes[f])
			{
				print("ProfileDataSet::save - failed to read profile [%d/%d]!\n", s + 1, m_n_examples);
				return false;
			}
		}
	}

	file.close();

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
