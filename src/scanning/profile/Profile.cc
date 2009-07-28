#include "Profile.h"
#include "GreedyExplorer.h"

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////
	// Functions to get the variation axis/planes for some index in the profile
	//////////////////////////////////////////////////////////////////////////

	bool isProfileOxys(const GreedyExplorer& explorer, int index)
	{
		return true;
	}

	bool isProfileOs(const GreedyExplorer& explorer, int index)
	{
		return explorer.getContextOx(index) == 0 && explorer.getContextOy(index) == 0;
	}

	bool isProfileOx(const GreedyExplorer& explorer, int index)
	{
		return explorer.getContextOy(index) == 0 && explorer.getContextOs(index) == 0;
	}

	bool isProfileOy(const GreedyExplorer& explorer, int index)
	{
		return explorer.getContextOx(index) == 0 && explorer.getContextOs(index) == 0;
	}

	bool isProfileOxy(const GreedyExplorer& explorer, int index)
	{
		return explorer.getContextOs(index) == 0;
	}

	bool isProfileOxs(const GreedyExplorer& explorer, int index)
	{
		return explorer.getContextOy(index) == 0;
	}

	bool isProfileOys(const GreedyExplorer& explorer, int index)
	{
		return explorer.getContextOx(index) == 0;
	}

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

/////////////////////////////////////////////////////////////////////////
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

void Profile::reset(const Pattern& pattern, const GreedyExplorer& explorer)
{
	m_pattern.copy(pattern);

	// Reset feature values
	for (int f = 0; f < NoFeatures; f ++)
	{
		m_features[f].fill(0.0);
	}

	// Functions to check axis indexes
	typedef bool (* FnCheckAxis)(const GreedyExplorer&, int);
	FnCheckAxis axis_checks[NoAxis] =
	{
		isProfileOxys,
		isProfileOx,
		isProfileOy,
		isProfileOs,
		isProfileOxy,
		isProfileOxs,
		isProfileOys
	};

	// Number of axis
	int axis_no[NoAxis] =
	{
		3, 1, 1, 1, 2, 2, 2
	};

	// Axis indexes
	typedef int (GreedyExplorer:: *FnGetAxis)(int) const;
	FnGetAxis axis_indexes[NoAxis][3] =
	{
		{ &GreedyExplorer::getContextOx, &GreedyExplorer::getContextOy, &GreedyExplorer::getContextOs },
		{ &GreedyExplorer::getContextOx, 0, 0 },
		{ &GreedyExplorer::getContextOy, 0, 0 },
		{ &GreedyExplorer::getContextOs, 0, 0 },
		{ &GreedyExplorer::getContextOx, &GreedyExplorer::getContextOy, 0 },
		{ &GreedyExplorer::getContextOx, &GreedyExplorer::getContextOs, 0 },
		{ &GreedyExplorer::getContextOy, &GreedyExplorer::getContextOs, 0 }
	};

	const unsigned char* ctx_flags = explorer.getContextFlags();
	const double* ctx_scores = explorer.getContextScores();
	const int ctx_size = explorer.getContextSize();

	// Extract feature values - counts & scores
	for (int a = All; a < NoAxis; a ++)
	{
		FnGetAxis* axis_indexes_a = axis_indexes[a];
		FnCheckAxis axis_check_a = axis_checks[a];
		const int axis_no_a = axis_no[a];

		// Counts
		for (int i = 0; i < ctx_size; i ++)
			if (ctx_flags[i] != 0x00 && axis_check_a(explorer, i) == true)
			{
				increment(Feature_Counts_All + a);
			}

		// Score amplitude - find the extreme scores for this axis
		double min_score = 10000.0, max_score = -10000.0;
		double sum_score = 0.0;
		int cnt = 0;
		for (int i = 0; i < ctx_size; i ++)
			if (ctx_flags[i] != 0x00 && axis_check_a(explorer, i) == true)
			{
				const double score = ctx_scores[i];
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

			for (int i = 0; i < ctx_size; i ++)
				if (ctx_flags[i] != 0x00 && axis_check_a(explorer, i) == true)
				{
					const double diff = ctx_scores[i] - avg_score;
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
		for (int i = 0; i < ctx_size; i ++)
			if (ctx_flags[i] != 0x00 && axis_check_a(explorer, i) == true)
			{
				for (int j = 0; j < axis_no_a; j ++)
				{
					const int axis = (explorer.*axis_indexes_a[j])(i);
					min_hits[j] = min(min_hits[j], axis);
					max_hits[j] = max(max_hits[j], axis);
					sum_hits[j] += axis;
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

				for (int i = 0; i < ctx_size; i ++)
					if (ctx_flags[i] != 0x00 && axis_check_a(explorer, i) == true)
					{
						const int axis = (explorer.*axis_indexes_a[j])(i);
						const double diff = axis - avg_hits;
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
