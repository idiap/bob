/**
 * @file cxx/old/scanning/src/Context.cc
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
#include "scanning/Context.h"
#include "scanning/ContextExplorer.h"

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////
	// Functions to get the variation axis/planes for some index in the context
	//////////////////////////////////////////////////////////////////////////

	bool isContextOxys(const ContextExplorer& explorer, int index)
	{
		return true;
	}

	bool isContextOs(const ContextExplorer& explorer, int index)
	{
		return explorer.getContextOx(index) == 0 && explorer.getContextOy(index) == 0;
	}

	bool isContextOx(const ContextExplorer& explorer, int index)
	{
		return explorer.getContextOy(index) == 0 && explorer.getContextOs(index) == 0;
	}

	bool isContextOy(const ContextExplorer& explorer, int index)
	{
		return explorer.getContextOx(index) == 0 && explorer.getContextOs(index) == 0;
	}

	bool isContextOxy(const ContextExplorer& explorer, int index)
	{
		return explorer.getContextOs(index) == 0;
	}

	bool isContextOxs(const ContextExplorer& explorer, int index)
	{
		return explorer.getContextOy(index) == 0;
	}

	bool isContextOys(const ContextExplorer& explorer, int index)
	{
		return explorer.getContextOx(index) == 0;
	}

/////////////////////////////////////////////////////////////////////////
// Constructor

Context::Context()
	:	m_features(new DoubleTensor[NoFeatures])
{
	for (int f = 0; f < NoFeatures; f ++)
	{
		m_features[f].resize(FeatureSizes[f]);
	}
}

/////////////////////////////////////////////////////////////////////////
// Destructor

Context::~Context()
{
	delete[] m_features;
}

/////////////////////////////////////////////////////////////////////////
// Copy constructor

Context::Context(const Context& other)
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

Context& Context::operator=(const Context& other)
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
// Reset to a new sub-window context

enum Axis
{
	All = 0,
	Ox, Oy, Os,
	Oxy, Oxs, Oys,
	NoAxis
};

void Context::reset(const Pattern& pattern, const ContextExplorer& explorer)
{
	m_pattern.copy(pattern);

	// Reset feature values
	for (int f = 0; f < NoFeatures; f ++)
	{
		m_features[f].fill(0.0);
	}

	// Functions to check axis indexes
	typedef bool (* FnCheckAxis)(const ContextExplorer&, int);
	FnCheckAxis axis_checks[NoAxis] =
	{
		isContextOxys,
		isContextOx,
		isContextOy,
		isContextOs,
		isContextOxy,
		isContextOxs,
		isContextOys
	};

	// Number of axis
	int axis_no[NoAxis] =
	{
		3, 1, 1, 1, 2, 2, 2
	};

	// Axis indexes
	typedef int (ContextExplorer:: *FnGetAxis)(int) const;
	FnGetAxis axis_indexes[NoAxis][3] =
	{
		{ &ContextExplorer::getContextOx, &ContextExplorer::getContextOy, &ContextExplorer::getContextOs },
		{ &ContextExplorer::getContextOx, 0, 0 },
		{ &ContextExplorer::getContextOy, 0, 0 },
		{ &ContextExplorer::getContextOs, 0, 0 },
		{ &ContextExplorer::getContextOx, &ContextExplorer::getContextOy, 0 },
		{ &ContextExplorer::getContextOx, &ContextExplorer::getContextOs, 0 },
		{ &ContextExplorer::getContextOy, &ContextExplorer::getContextOs, 0 }
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
				min_score = std::min(min_score, score);
				max_score = std::max(max_score, score);
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
					min_hits[j] = std::min(min_hits[j], axis);
					max_hits[j] = std::max(max_hits[j], axis);
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

void Context::copyTo(DoubleTensor& tensor) const
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

bool Context::copyFrom(const DoubleTensor& tensor)
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
