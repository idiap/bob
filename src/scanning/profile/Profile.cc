#include "Profile.h"
#include "GreedyExplorer.h"

namespace Torch
{
namespace Private
{
	//////////////////////////////////////////////////////////////////////////
	// Functions to get the variation axis/planes for some index in the profile
	//////////////////////////////////////////////////////////////////////////

	const int ProfileSize =	Torch::GreedyExplorer::NoConfigs;

	struct ProfileIndexes
	{
	private:
		// Constructor
		ProfileIndexes()
			:	m_ox(new int[ProfileSize]),
				m_oy(new int[ProfileSize]),
				m_os(new int[ProfileSize])
		{

			// Initialize 3D axis coordinates for each profile linear index
			int index = 0;
			for (int is = -Torch::GreedyExplorer::NoVarS; is <= Torch::GreedyExplorer::NoVarS; is ++)
			{
				for (int ix = -Torch::GreedyExplorer::NoVarX; ix <= Torch::GreedyExplorer::NoVarX; ix ++)
				{
					for (int iy = -Torch::GreedyExplorer::NoVarY; iy <= Torch::GreedyExplorer::NoVarY; iy ++)
					{
						m_ox[index] = ix;
						m_oy[index] = iy;
						m_os[index] = is;
						index ++;
					}
				}
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

	bool isProfileOxys(int index)
	{
		return true;
	}

	bool isProfileOs(int index)
	{
		static const ProfileIndexes& pfi = ProfileIndexes::getInstance();
		return pfi.m_ox[index] == 0 && pfi.m_oy[index] == 0;
	}

	bool isProfileOx(int index)
	{
		static const ProfileIndexes& pfi = ProfileIndexes::getInstance();
		return pfi.m_oy[index] == 0 && pfi.m_os[index] == 0;
	}

	bool isProfileOy(int index)
	{
		static const ProfileIndexes& pfi = ProfileIndexes::getInstance();
		return pfi.m_ox[index] == 0 && pfi.m_os[index] == 0;
	}

	bool isProfileOxy(int index)
	{
		static const ProfileIndexes& pfi = ProfileIndexes::getInstance();
		return pfi.m_os[index] == 0;
	}

	bool isProfileOxs(int index)
	{
		static const ProfileIndexes& pfi = ProfileIndexes::getInstance();
		return pfi.m_oy[index] == 0;
	}

	bool isProfileOys(int index)
	{
		static const ProfileIndexes& pfi = ProfileIndexes::getInstance();
		return pfi.m_ox[index] == 0;
	}
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

enum Axis
{
	All = 0,
	Ox, Oy, Os,
	Oxy, Oxs, Oys,
	NoAxis
};

void Profile::reset(const Pattern& pattern, const unsigned char* pf_flags, const double* pf_scores)
{
	m_pattern.copy(pattern);

	// Reset feature values
	for (int f = 0; f < NoFeatures; f ++)
	{
		m_features[f].fill(0.0);
	}

	// Functions to check axis indexes
	typedef bool (* FnCheckAxis)(int);
	FnCheckAxis axis_checks[NoAxis] =
	{
		Private::isProfileOxys,
		Private::isProfileOx,
		Private::isProfileOy,
		Private::isProfileOs,
		Private::isProfileOxy,
		Private::isProfileOxs,
		Private::isProfileOys
	};

	// Number of axis
	int axis_no[NoAxis] =
	{
		3, 1, 1, 1, 2, 2, 2
	};

	// Axis indexes
	static const Private::ProfileIndexes& pfi = Private::ProfileIndexes::getInstance();
	int* axis_indexes[NoAxis][3] =
	{
		{ pfi.m_ox, pfi.m_oy, pfi.m_os },
		{ pfi.m_ox, 0, 0 },
		{ pfi.m_oy, 0, 0 },
		{ pfi.m_os, 0, 0 },
		{ pfi.m_ox, pfi.m_oy, 0 },
		{ pfi.m_ox, pfi.m_os, 0 },
		{ pfi.m_oy, pfi.m_os, 0 }
	};

	// Extract feature values - counts & scores
	for (int a = All; a < NoAxis; a ++)
	{
		int** axis_indexes_a = axis_indexes[a];
		FnCheckAxis axis_check_a = axis_checks[a];
		const int axis_no_a = axis_no[a];

		// Counts
		for (int i = 0; i < Private::ProfileSize; i ++)
			if (pf_flags[i] != 0x00 && axis_check_a(i) == true)
			{
				increment(Feature_Counts_All + a);
			}

		// Score amplitude - find the extreme scores for this axis
		double min_score = 10000.0, max_score = -10000.0;
		double sum_score = 0.0;
		int cnt = 0;
		for (int i = 0; i < Private::ProfileSize; i ++)
			if (pf_flags[i] != 0x00 && axis_check_a(i) == true)
			{
				const double score = pf_scores[i];
				min_score = min(min_score, score);
				max_score = max(max_score, score);
				sum_score += score;
				cnt ++;
			}

		// Score amplitude - set the value
		if (cnt > 0)
		{
			m_features[Feature_Scores_Ampl_All + a].set(0, max_score - min_score);
		}

		// Score standard deviation - set the value
		if (cnt > 1)
		{
			const double avg_score = sum_score / cnt;
			sum_score = 0.0;

			for (int i = 0; i < Private::ProfileSize; i ++)
				if (pf_flags[i] != 0x00 && axis_check_a(i) == true)
				{
					const double diff = pf_scores[i] - avg_score;
					sum_score += diff * diff;
				}

			m_features[Feature_Scores_Stdev_All + a].set(0, sqrt(sum_score / cnt));
		}

		// Hit amplitude - find the extreme hits for this axis
		int min_hits[3], max_hits[3];
		double sum_hits[3];
		for (int j = 0; j < axis_no_a; j ++)
		{
			min_hits[j] = 10000;
			max_hits[j] = -10000;
			sum_hits[j] = 0.0;
		}

		cnt = 0;
		for (int i = 0; i < Private::ProfileSize; i ++)
			if (pf_flags[i] != 0x00 && axis_check_a(i) == true)
			{
				for (int j = 0; j < axis_no_a; j ++)
				{
					min_hits[j] = min(min_hits[j], axis_indexes_a[j][i]);
					max_hits[j] = max(max_hits[j], axis_indexes_a[j][i]);
					sum_hits[j] += axis_indexes_a[j][i];
				}

				cnt ++;
			}

		// Hit amplitude - set the value
		if (cnt > 0)
		{
			for (int j = 0; j < axis_no_a; j ++)
			{
				m_features[Feature_Hits_Ampl_All + a].set(j, max_hits[j] - min_hits[j]);
			}
		}

		// Hit standard deviation - set the value
		if (cnt > 0)
		{
			for (int j = 0; j < axis_no_a; j ++)
			{
				const double avg_hits = sum_hits[j] / cnt;
				sum_hits[j] = 0.0;

				for (int i = 0; i < Private::ProfileSize; i ++)
					if (pf_flags[i] != 0x00 && axis_check_a(i) == true)
					{
						const double diff = axis_indexes_a[j][i] - avg_hits;
						sum_hits[j] += diff * diff;
					}

				m_features[Feature_Hits_Stdev_All + a].set(j, sqrt(sum_hits[j] / cnt));
			}
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
