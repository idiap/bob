#include "Profile.h"
#include "GreedyExplorer.h"

namespace Torch
{
namespace Private
{
	//////////////////////////////////////////////////////////////////////////
	// Functions to get the variation axis/planes for some index in the profile
	//////////////////////////////////////////////////////////////////////////

	const int ProfileSize =	(2 * Torch::GreedyExplorer::NoVarX + 1) *
				(2 * Torch::GreedyExplorer::NoVarY + 1) *
				(2 * Torch::GreedyExplorer::NoVarS + 1);

	const int base_s = (2 * Torch::GreedyExplorer::NoVarX + 1) * (2 * Torch::GreedyExplorer::NoVarY + 1);
	const int base_x = (2 * Torch::GreedyExplorer::NoVarY + 1);
	const int base_y = 1;

	int getOsIndex(int index) { return index / base_s; }
	int getOxIndex(int index) { return (index - getOsIndex(index) * base_s) / base_x; }
	int getOyIndex(int index) { return (index - getOsIndex(index) * base_s - getOxIndex(index) * base_x) / base_y; }

	bool isProfileOxys(int index)
	{
		return true;
	}

	bool isProfileOs(int index)
	{
		return 	getOxIndex(index) == GreedyExplorer::NoVarX &&
			getOyIndex(index) == GreedyExplorer::NoVarY;
	}

	bool isProfileOx(int index)
	{
		return 	getOsIndex(index) == GreedyExplorer::NoVarS &&
			getOyIndex(index) == GreedyExplorer::NoVarY;
	}

	bool isProfileOy(int index)
	{
		return 	getOsIndex(index) == GreedyExplorer::NoVarS &&
			getOxIndex(index) == GreedyExplorer::NoVarX;
	}

	bool isProfileOxy(int index)
	{
		return 	getOsIndex(index) == GreedyExplorer::NoVarS;
	}

	bool isProfileOxs(int index)
	{
		return 	getOyIndex(index) == GreedyExplorer::NoVarY;
	}

	bool isProfileOys(int index)
	{
		return 	getOxIndex(index) == GreedyExplorer::NoVarX;
	}

	struct ProfileIndexes
	{
	private:
		// Constructor
		ProfileIndexes()
			:	m_ox(new int[ProfileSize]),
				m_oy(new int[ProfileSize]),
				m_os(new int[ProfileSize])
		{
			for (int i = 0; i < ProfileSize; i ++)
			{
				m_ox[i] = getOxIndex(i) - GreedyExplorer::NoVarX;
				m_oy[i] = getOyIndex(i) - GreedyExplorer::NoVarY;
				m_os[i] = getOsIndex(i) - GreedyExplorer::NoVarS;
			}
		}

	public:

		// Destructor
		~ProfileIndexes()
		{
			delete[] m_ox;
			delete[] m_oy;
			delete[] m_os;
		}

		// Get the only instance
		static const ProfileIndexes& getInstance()
		{
			static const ProfileIndexes pf_indexes;
			return pf_indexes;
		}

		/////////////////////////////////////////////////////////////
		// Attributes

		int*			m_ox;	// [ProfileSize], <Ox> indexes from the linear index
		int*			m_oy;	// [ProfileSize], <Oy> indexes from the linear index
		int*			m_os;	// [ProfileSize], <Os> indexes from the linear index
	};
}// End of Private

/////////////////////////////////////////////////////////////////////////
// Constructor

Profile::Profile()
	:	m_features(new DoubleTensor[NoFeatures])
{
	for (int f = 0; f < NoFeatures; f ++)
	{
		m_features[f].resize(FeatureSizes[f]);
	}
}

/////////////////////////////////////////////////////////////////////////
// Destructor

Profile::~Profile()
{
	delete[] m_features;
}

/////////////////////////////////////////////////////////////////////////
// Copy constructor

Profile::Profile(const Profile& other)
	:	m_features(new DoubleTensor[NoFeatures])
{
	for (int f = 0; f < NoFeatures; f ++)
	{
		m_features[f].resize(FeatureSizes[f]);
	}
	operator=(other);
}

////////////////homes/catana/coding/torch5spro/src/scanning/profile/ProfileMachine.h:65://////////////////////////////////////////////////////////
// Assignment operator

Profile& Profile::operator=(const Profile& other)
{
	for (int f = 0; f < NoFeatures; f ++)
	{
		const int fsize = FeatureSizes[f];
		for (int i = 0; i < fsize; i ++)
		{
			m_features[f].set(i, other.m_features[f].get(i));
		}
	}

	return *this;
}

/////////////////////////////////////////////////////////////////////////
// Reset to a new sub-window profile

void Profile::reset(const Pattern& pattern, const unsigned char* pf_flags, const double* pf_scores)
{
	static const Private::ProfileIndexes& pfi = Private::ProfileIndexes::getInstance();

	m_pattern.copy(pattern);

	// Reset some features
	for (int f = Feature_Counts_All; f <= Feature_Counts_Oys; f ++)
	{
		m_features[f].fill(0.0);
	}
	for (int f = Feature_Scores_MinMax_All; f <= Feature_Scores_MinMax_Oys; f ++)
	{
		m_features[f].set(0, +100.0);
		m_features[f].set(1, -100.0);
	}
	for (int f = Feature_Hits_Diff_All; f <= Feature_Hits_Diff_Oys; f ++)
	{
		m_features[f].fill(0.0);
	}
		// Hits minmax: extreme values for each axis
	m_features[Feature_Hits_MinMax_All].set(0, GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_All].set(1, -GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_All].set(2, GreedyExplorer::NoVarY);
	m_features[Feature_Hits_MinMax_All].set(3, -GreedyExplorer::NoVarY);
	m_features[Feature_Hits_MinMax_All].set(4, GreedyExplorer::NoVarS);
	m_features[Feature_Hits_MinMax_All].set(5, -GreedyExplorer::NoVarS);

	m_features[Feature_Hits_MinMax_Ox].set(0, GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_Ox].set(1, -GreedyExplorer::NoVarX);

	m_features[Feature_Hits_MinMax_Oy].set(0, GreedyExplorer::NoVarY);
	m_features[Feature_Hits_MinMax_Oy].set(1, -GreedyExplorer::NoVarY);

	m_features[Feature_Hits_MinMax_Os].set(0, GreedyExplorer::NoVarS);
	m_features[Feature_Hits_MinMax_Os].set(1, -GreedyExplorer::NoVarS);

	m_features[Feature_Hits_MinMax_Oxy].set(0, GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_Oxy].set(1, -GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_Oxy].set(2, GreedyExplorer::NoVarY);
	m_features[Feature_Hits_MinMax_Oxy].set(3, -GreedyExplorer::NoVarY);

	m_features[Feature_Hits_MinMax_Oxs].set(0, GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_Oxs].set(1, -GreedyExplorer::NoVarX);
	m_features[Feature_Hits_MinMax_Oxs].set(2, GreedyExplorer::NoVarS);
	m_features[Feature_Hits_MinMax_Oxs].set(3, -GreedyExplorer::NoVarS);

	m_features[Feature_Hits_MinMax_Oys].set(0, GreedyExplorer::NoVarY);
	m_features[Feature_Hits_MinMax_Oys].set(1, -GreedyExplorer::NoVarY);
	m_features[Feature_Hits_MinMax_Oys].set(2, GreedyExplorer::NoVarS);
	m_features[Feature_Hits_MinMax_Oys].set(3, -GreedyExplorer::NoVarS);

	// Extract features from each detection in the profile
	for (int i = 0; i < Private::ProfileSize; i ++)
		if (pf_flags[i] != 0x00)
	{
		const int ox = pfi.m_ox[i];
		const int oy = pfi.m_oy[i];
		const int os = pfi.m_os[i];
		const double score = pf_scores[i];

		// Features that use 3 axis
		increment(Feature_Counts_All);
		update_min(Feature_Scores_MinMax_All, 0, score);
		update_max(Feature_Scores_MinMax_All, 1, score);

		update_min(Feature_Hits_MinMax_All, 0, ox);
		update_max(Feature_Hits_MinMax_All, 1, ox);
		update_min(Feature_Hits_MinMax_All, 2, oy);
		update_max(Feature_Hits_MinMax_All, 3, oy);
		update_min(Feature_Hits_MinMax_All, 4, os);
		update_max(Feature_Hits_MinMax_All, 5, os);

		update_dif(Feature_Hits_Diff_All, 0, Feature_Hits_MinMax_All, 1, 0);
		update_dif(Feature_Hits_Diff_All, 1, Feature_Hits_MinMax_All, 3, 2);
		update_dif(Feature_Hits_Diff_All, 2, Feature_Hits_MinMax_All, 5, 4);

		// Features that use 1 axis
		if (Private::isProfileOx(i) == true)
		{
			// Ox
			increment(Feature_Counts_Ox);
			update_min(Feature_Scores_MinMax_Ox, 0, score);
			update_max(Feature_Scores_MinMax_Ox, 1, score);

			update_min(Feature_Hits_MinMax_Ox, 0, ox);
			update_max(Feature_Hits_MinMax_Ox, 1, ox);

			update_dif(Feature_Hits_Diff_Ox, 0, Feature_Hits_MinMax_Ox, 1, 0);
		}
		if (Private::isProfileOy(i) == true)
		{
			// Oy
			increment(Feature_Counts_Oy);
			update_min(Feature_Scores_MinMax_Oy, 0, score);
			update_max(Feature_Scores_MinMax_Oy, 1, score);

			update_min(Feature_Hits_MinMax_Oy, 0, oy);
			update_max(Feature_Hits_MinMax_Oy, 1, oy);

			update_dif(Feature_Hits_Diff_Oy, 0, Feature_Hits_MinMax_Oy, 1, 0);
		}
		if (Private::isProfileOs(i) == true)
		{
			// Os
			increment(Feature_Counts_Os);
			update_min(Feature_Scores_MinMax_Os, 0, score);
			update_max(Feature_Scores_MinMax_Os, 1, score);

			update_min(Feature_Hits_MinMax_Os, 0, os);
			update_max(Feature_Hits_MinMax_Os, 1, os);

			update_dif(Feature_Hits_Diff_Os, 0, Feature_Hits_MinMax_Os, 1, 0);
		}

		// Features that use 2 axis
		if (Private::isProfileOxy(i) == true)
		{
			// Oxy
			increment(Feature_Counts_Oxy);
			update_min(Feature_Scores_MinMax_Oxy, 0, score);
			update_max(Feature_Scores_MinMax_Oxy, 1, score);

			update_min(Feature_Hits_MinMax_Oxy, 0, ox);
			update_max(Feature_Hits_MinMax_Oxy, 1, ox);
			update_min(Feature_Hits_MinMax_Oxy, 2, oy);
			update_max(Feature_Hits_MinMax_Oxy, 3, oy);

			update_dif(Feature_Hits_Diff_Oxy, 0, Feature_Hits_MinMax_Oxy, 1, 0);
			update_dif(Feature_Hits_Diff_Oxy, 1, Feature_Hits_MinMax_Oxy, 3, 2);
		}
		if (Private::isProfileOxs(i) == true)
		{
			// Oxs
			increment(Feature_Counts_Oxs);
			update_min(Feature_Scores_MinMax_Oxs, 0, score);
			update_max(Feature_Scores_MinMax_Oxs, 1, score);

			update_min(Feature_Hits_MinMax_Oxs, 0, ox);
			update_max(Feature_Hits_MinMax_Oxs, 1, ox);
			update_min(Feature_Hits_MinMax_Oxs, 2, os);
			update_max(Feature_Hits_MinMax_Oxs, 3, os);

			update_dif(Feature_Hits_Diff_Oxs, 0, Feature_Hits_MinMax_Oxs, 1, 0);
			update_dif(Feature_Hits_Diff_Oxs, 1, Feature_Hits_MinMax_Oxs, 3, 2);
		}
		if (Private::isProfileOys(i) == true)
		{
			// Oys
			increment(Feature_Counts_Oys);
			update_min(Feature_Scores_MinMax_Oys, 0, score);
			update_max(Feature_Scores_MinMax_Oys, 1, score);

			update_min(Feature_Hits_MinMax_Oys, 0, oy);
			update_max(Feature_Hits_MinMax_Oys, 1, oy);
			update_min(Feature_Hits_MinMax_Oys, 2, os);
			update_max(Feature_Hits_MinMax_Oys, 3, os);

			update_dif(Feature_Hits_Diff_Oys, 0, Feature_Hits_MinMax_Oys, 1, 0);
			update_dif(Feature_Hits_Diff_Oys, 1, Feature_Hits_MinMax_Oys, 3, 2);
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Copy the features to 2D tensor (to be passed to machines) of [NoFeatures] x [MaxNoDimensions]

void Profile::copyTo(DoubleTensor& tensor) const
{
	tensor.resize(NoFeatures, MaxNoDimensions);

	for (int f = 0; f < NoFeatures; f ++)
	{
		const int fsize = FeatureSizes[f];
		for (int i = 0; i < fsize; i ++)
		{
			tensor.set(f, i, m_features[f].get(i));
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Copy the features from a 2D tensor of [NoFeatures] x [MaxNoDimensions]

bool Profile::copyFrom(const DoubleTensor& tensor)
{
	if (	tensor.nDimension() != 2 ||
		tensor.size(0) != NoFeatures ||
		tensor.size(1) != MaxNoDimensions)
	{
		return false;
	}

	for (int f = 0; f < NoFeatures; f ++)
	{
		const int fsize = FeatureSizes[f];
		for (int i = 0; i < fsize; i ++)
		{
			m_features[f].set(i, tensor.get(f, i));
		}
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////

}
