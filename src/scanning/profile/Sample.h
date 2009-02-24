#ifndef _TORCHVISION_SCANNING_PROFILE_SAMPLE_H_
#define _TORCHVISION_SCANNING_PROFILE_SAMPLE_H_

namespace Torch
{
namespace Profile
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

		Feature_Scores_All,
		Feature_Scores_Ox,
		Feature_Scores_Oy,
		Feature_Scores_Os,
		Feature_Scores_Oxy,
		Feature_Scores_Oxs,
		Feature_Scores_Oys,

		Feature_Hits_All,
		Feature_Hits_Ox,
		Feature_Hits_Oy,
		Feature_Hits_Os,
		Feature_Hits_Oxy,
		Feature_Hits_Oxs,
		Feature_Hits_Oys,

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
	const char FeatureNames[NoFeatures][16] =
	{
		"counts_all", "counts_ox", "counts_oy", "counts_os", "counts_oxy", "counts_oxs", "counts_oys",
		"scores_all", "scores_ox", "scores_oy", "scores_os", "scores_oxy", "scores_oxs", "scores_oys",
		"hits_all", "hits_ox", "hits_oy", "hits_os", "hits_oxy", "hits_oxs", "hits_oys",
		"hits_minmax_all", "hits_minmax_ox", "hits_minmax_oy", "hits_minmax_os", "hits_minmax_oxy", "hits_minmax_oxs", "hits_minmax_oys",
		"hits_diff_all", "hits_diff_ox", "hits_diff_oy", "hits_diff_os", "hits_diff_oxy", "hits_diff_oxs", "hits_diff_oys"
	};

	const int FeatureSizes[NoFeatures] =
	{
		1, 1, 1, 1, 1, 1, 1,	// Counts - 1D data
		1, 1, 1, 1, 1, 1, 1,	// Scores - 1D data
		3, 1, 1, 1, 2, 2, 2,	// Hits - 1-3D data
		6, 2, 2, 2, 4, 4, 4,	// Hits + min-max - 2-6D data
		3, 1, 1, 1, 2, 2, 2	// Hits + diff - 1-3D data
	};

	// Feature extracted from some profile
	struct Feature
	{
		static const int MaxSize = 6;

		// Constructors
		Feature(double value0 = 0.0)
			: 	m_data(new double[MaxSize])
		{
			m_data[0] = value0;
			m_data[1] = 0.0;
			m_data[2] = 0.0;
			m_data[3] = 0.0;
			m_data[4] = 0.0;
			m_data[5] = 0.0;
		}
		Feature(double value0, double value1)
			: 	m_data(new double[MaxSize])
		{
			m_data[0] = value0;
			m_data[1] = value1;
			m_data[2] = 0.0;
			m_data[3] = 0.0;
			m_data[4] = 0.0;
			m_data[5] = 0.0;
		}
		Feature(double value0, double value1, double value2)
			: 	m_data(new double[MaxSize])
		{
			m_data[0] = value0;
			m_data[1] = value1;
			m_data[2] = value2;
			m_data[3] = 0.0;
			m_data[4] = 0.0;
			m_data[5] = 0.0;
		}
		Feature(double value0, double value1, double value2, double value3)
			: 	m_data(new double[MaxSize])
		{
			m_data[0] = value0;
			m_data[1] = value1;
			m_data[2] = value2;
			m_data[3] = value3;
			m_data[4] = 0.0;
			m_data[5] = 0.0;
		}
		Feature(double value0, double value1, double value2, double value3, double value4)
			: 	m_data(new double[MaxSize])
		{
			m_data[0] = value0;
			m_data[1] = value1;
			m_data[2] = value2;
			m_data[3] = value3;
			m_data[4] = value4;
			m_data[5] = 0.0;
		}
		Feature(double value0, double value1, double value2, double value3, double value4, double value5)
			: 	m_data(new double[MaxSize])
		{
			m_data[0] = value0;
			m_data[1] = value1;
			m_data[2] = value2;
			m_data[3] = value3;
			m_data[4] = value4;
			m_data[5] = value5;
		}
		Feature(const Feature& other)
			: 	m_data(new double[MaxSize])
		{
			operator=(other);
		}

		// Assignment operator
		Feature& operator=(const Feature& other)
		{
			for (int i = 0; i < MaxSize; i ++)
			{
				m_data[i] = other.m_data[i];
			}
		}

		// Destructor
		~Feature()
		{
			delete[] m_data;
		}

		// Attributes
		double*			m_data;			// Maximum 6D data
	};

        /////////////////////////////////////////////////////////////////////////
	// Torch::Profile::Sample:
	//	- gathers features from profile data (around some sub-window)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct Sample
	{
		// Constructor
		Sample();

		/*// Delete collected features
		void 			clear();

		// Cumulate a new detection
		void			cumulate(int index, double score);

		/////////////////////////////////////////////////////////////
		// Attributes

		Torch::Pattern		m_pattern;		// Sub-window seed
		std::vector<std::vector<Feature> > m_features;	// [NoFeatures]: Extracted features
		*/
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::Profile::Distribution:
	//	- computes the pos/neg/gt feature distribution
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct Distribution
	{
		// Constructor
		Distribution();

		/*
		// Clear cumulated profiles
		void			clear();

		// Cumulate a new profile (negative, positive or ground truth)
		void			cumulate(bool positive, const Example& profile);
		void			cumulate(const Example& gt_profile);

		// Save the distribution
		void			save(const char* dir_data, const char* name) const;

		/////////////////////////////////////////////////////////////
		// Attributes

		std::vector<Example>	m_neg_examples;		// Negative examples
		std::vector<Example>	m_pos_examples;		// Positive examples
		std::vector<Example>	m_gt_examples;		// Ground truth examples
		*/
	};
}
}

#endif
