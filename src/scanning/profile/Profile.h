#ifndef _TORCHVISION_SCANNING_PROFILE_SAMPLE_H_
#define _TORCHVISION_SCANNING_PROFILE_SAMPLE_H_

#include "Pattern.h"
#include "Tensor.h"

namespace Torch
{
	// Features
	enum FeatureType
	{
		Feature_Counts_All = 0,
		Feature_Counts_Ox,
		Feature_Counts_Oy,
		Feature_Counts_Os,
		Feature_Counts_Oxy,
		Feature_Counts_Oxs,
		Feature_Counts_Oys,

		Feature_Scores_MinMax_All,
		Feature_Scores_MinMax_Ox,
		Feature_Scores_MinMax_Oy,
		Feature_Scores_MinMax_Os,
		Feature_Scores_MinMax_Oxy,
		Feature_Scores_MinMax_Oxs,
		Feature_Scores_MinMax_Oys,

		Feature_Hits_MinMax_All,
		Feature_Hits_MinMax_Ox,
		Feature_Hits_MinMax_Oy,
		Feature_Hits_MinMax_Os,
		Feature_Hits_MinMax_Oxy,
		Feature_Hits_MinMax_Oxs,
		Feature_Hits_MinMax_Oys,

		Feature_Hits_Diff_All,
		Feature_Hits_Diff_Ox,
		Feature_Hits_Diff_Oy,
		Feature_Hits_Diff_Os,
		Feature_Hits_Diff_Oxy,
		Feature_Hits_Diff_Oxs,
		Feature_Hits_Diff_Oys,

		NoFeatures
	};

	const char FeatureNames[NoFeatures][24] =
	{
		"counts_all",
		"counts_ox", "counts_oy", "counts_os",
		"counts_oxy", "counts_oxs", "counts_oys",

		"scores_minmax_all",
		"scores_minmax_ox", "scores_minmax_oy", "scores_minmax_os",
		"scores_minmax_oxy", "scores_minmax_oxs", "scores_minmax_oys",

		"hits_minmax_all",
		"hits_minmax_ox", "hits_minmax_oy", "hits_minmax_os",
		"hits_minmax_oxy", "hits_minmax_oxs", "hits_minmax_oys",

		"hits_diff_all",
		"hits_diff_ox", "hits_diff_oy", "hits_diff_os",
		"hits_diff_oxy", "hits_diff_oxs", "hits_diff_oys"
	};

	const int FeatureSizes[NoFeatures] =
	{
		1, 1, 1, 1, 1, 1, 1,	// Counts - 1D data
		2, 2, 2, 2, 2, 2, 2,	// Scores + min-max - 2D data
		6, 2, 2, 2, 4, 4, 4,	// Hits + min-max - 2-6D data
		3, 1, 1, 1, 2, 2, 2	// Hits + diff - 1-3D data
	};

        /////////////////////////////////////////////////////////////////////////
	// Torch::Sample:
	//	- gathers features from profile data (around some sub-window)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct Profile
	{
	public:
		// Constructor
		Profile();

		// Destructor
		~Profile();

		// Copy constructor
		Profile(const Profile& other);

		// Assignment operator
		Profile& operator=(const Profile& other);

		// Reset to a new sub-window profile
		void 			reset(	const Pattern& pattern,
						const unsigned char* pf_flags,
						const double* pf_scores);

		// Copy the features to 2D tensor (to be passed to machines) of [NoFeatures] x [MaxNoDimensions]
		void			copyTo(DoubleTensor& tensor) const;

		// Copy the features from a 2D tensor of [NoFeatures] x [MaxNoDimensions]
		bool			copyFrom(const DoubleTensor& tensor);

		// Helper functions
		void 			increment(int f)
		{
			m_features[f].set(0, m_features[f].get(0) + 1.0);
		}
		void 			update_max(int f, int index, double value)
		{
			m_features[f].set(index, max(m_features[f].get(index), value));
		}
		void 			update_min(int f, int index, double value)
		{
			m_features[f].set(index, min(m_features[f].get(index), value));
		}
		void 			update_dif(int f, int index, int f2, int index21, int index22)
		{
			m_features[f].set(index, m_features[f2].get(index21) - m_features[f2].get(index22));
		}

		/////////////////////////////////////////////////////////////
		// Attributes

		static const int 	MaxNoDimensions = 6;

		Pattern			m_pattern;		// Sub-window seed
		DoubleTensor*		m_features;		// [NoFeatures] - Extracted feature values
	};
}

#endif
