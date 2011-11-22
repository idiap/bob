/**
 * @file cxx/old/scanning/scanning/Context.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCHVISION_SCANNING_CONTEXT_SAMPLE_H_
#define _TORCHVISION_SCANNING_CONTEXT_SAMPLE_H_

#include "scanning/Pattern.h"
#include "core/Tensor.h"

namespace Torch
{
	class ContextExplorer;

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

		Feature_Scores_Ampl_All,
		Feature_Scores_Ampl_Ox,
		Feature_Scores_Ampl_Oy,
		Feature_Scores_Ampl_Os,
		Feature_Scores_Ampl_Oxy,
		Feature_Scores_Ampl_Oxs,
		Feature_Scores_Ampl_Oys,

		Feature_Scores_Stdev_All,
		Feature_Scores_Stdev_Ox,
		Feature_Scores_Stdev_Oy,
		Feature_Scores_Stdev_Os,
		Feature_Scores_Stdev_Oxy,
		Feature_Scores_Stdev_Oxs,
		Feature_Scores_Stdev_Oys,

		Feature_Hits_Ampl_All,
		Feature_Hits_Ampl_Ox,
		Feature_Hits_Ampl_Oy,
		Feature_Hits_Ampl_Os,
		Feature_Hits_Ampl_Oxy,
		Feature_Hits_Ampl_Oxs,
		Feature_Hits_Ampl_Oys,

		Feature_Hits_Stdev_All,
		Feature_Hits_Stdev_Ox,
		Feature_Hits_Stdev_Oy,
		Feature_Hits_Stdev_Os,
		Feature_Hits_Stdev_Oxy,
		Feature_Hits_Stdev_Oxs,
		Feature_Hits_Stdev_Oys,

		NoFeatures
	};

	const char FeatureNames[NoFeatures][24] =
	{
		"counts_all",
		"counts_ox", "counts_oy", "counts_os",
		"counts_oxy", "counts_oxs", "counts_oys",

		"scores_ampl_all",
		"scores_ampl_ox", "scores_ampl_oy", "scores_ampl_os",
		"scores_ampl_oxy", "scores_ampl_oxs", "scores_ampl_oys",

		"scores_stdev_all",
		"scores_stdev_ox", "scores_stdev_oy", "scores_stdev_os",
		"scores_stdev_oxy", "scores_stdev_oxs", "scores_stdev_oys",

		"hits_ampl_all",
		"hits_ampl_ox", "hits_ampl_oy", "hits_ampl_os",
		"hits_ampl_oxy", "hits_ampl_oxs", "hits_ampl_oys",

		"hits_stdev_all",
		"hits_stdev_ox", "hits_stdev_oy", "hits_stdev_os",
		"hits_stdev_oxy", "hits_stdev_oxs", "hits_stdev_oys"
	};

	const int FeatureSizes[NoFeatures] =
	{
		1, 1, 1, 1, 1, 1, 1,	// Counts - 1D data
		1, 1, 1, 1, 1, 1, 1,	// Score amplitude - 1D data
		1, 1, 1, 1, 1, 1, 1,	// Score standard deviation - 1D data
		3, 1, 1, 1, 2, 2, 2,	// Hits amplitude - 1-3D data
		3, 1, 1, 1, 2, 2, 2	// Hits standard deviation - 1-3D data
	};

        /////////////////////////////////////////////////////////////////////////
	// Torch::Context:
	//	- gathers features from context data (around some sub-window)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct Context
	{
	public:
		// Constructor
		Context();

		// Destructor
		~Context();

		// Copy constructor
		Context(const Context& other);

		// Assignment operator
		Context& operator=(const Context& other);

		// Reset to a new sub-window context
		void 			reset(	const Pattern& pattern,
						const ContextExplorer& explorer);

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
			m_features[f].set(index, std::max(m_features[f].get(index), value));
		}
		void 			update_min(int f, int index, double value)
		{
			m_features[f].set(index, std::min(m_features[f].get(index), value));
		}
		void 			update_dif(int f, int index, int f2, int index21, int index22)
		{
			m_features[f].set(index, m_features[f2].get(index21) - m_features[f2].get(index22));
		}

		/////////////////////////////////////////////////////////////
		// Attributes

		static const int 	MaxNoDimensions = 3;

		Pattern			m_pattern;		// Sub-window seed
		DoubleTensor*		m_features;		// [NoFeatures] - Extracted feature values
	};
}

#endif
