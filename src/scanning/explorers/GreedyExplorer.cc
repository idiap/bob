#include "GreedyExplorer.h"
#include "ipSWEvaluator.h"
#include "ScaleExplorer.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

GreedyExplorer::GreedyExplorer(ipSWEvaluator* swEvaluator, Mode mode)
	: 	MSExplorer(swEvaluator),
		m_mode(mode),
		m_profileFlags(new unsigned char[NoConfigs]),
		m_profileScores(new double[NoConfigs])
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

GreedyExplorer::~GreedyExplorer()
{
	delete[] m_profileFlags;
	delete[] m_profileScores;
}

/////////////////////////////////////////////////////////////////////////
// called when some option was changed - overriden

void GreedyExplorer::optionChanged(const char* name)
{
	MSExplorer::optionChanged(name);
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool GreedyExplorer::process()
{
	const bool verbose = getBOption("verbose");

	// Don't want to stop at the first iteration or to scale from large to small windows
	setBOption("StopAtFirstDetection", false);
	setBOption("StartWithLargeScales", false);

	// Scan the image using MS
	if (MSExplorer::process() == false)
	{
		Torch::message("GreedyExplorer::process - error initializing the search space!\n");
		return false;
	}

	// Check the working mode ...
	switch (m_mode)
	{
		// Object detection
	case Scanning:
		{
			/*
			const PatternList& lpatterns = m_clusterAlgo.getPatterns();
			const int n_patterns = lpatterns.size();

			// Cluster the SWs generated from MSExplorer
			m_clusterAlgo.clear();
			if (m_clusterAlgo.process(m_data->m_patterns) == false)
			{
				Torch::message("GreedyExplorer::process - error clustering SWs!\n");
				return false;
			}

			// Refine the search around the best points til it's possible
			const int old_n_candidates = m_data->m_patterns.size();
			while (true)
			{
				if (refineSearch() == false)
				{
					Torch::message("GreedyExplorer::process - error refining the search space!\n");
					return false;
				}

				// Debug message
				if (verbose == true)
				{
					Torch::print("[GreedyExplorer]: pruned = %d, scanned = %d, accepted = %d\n",
							m_data->m_stat_prunned,
							m_data->m_stat_scanned,
							m_data->m_stat_accepted);
				}

				// Check the stopping criterion
				if (shouldSearchMode(old_n_candidates) == false)
				{
					break;

					// Debug message
					if (verbose == true)
					{
						Torch::print("[GreedyExplorer]: stopping ...\n");
					}
				}
			}
			*/
		}
		break;

		// Profiling along candidate SWs
	case Profiling:
		{
			// Nothing to do - the user should retrieve the profiles by <profileSW>!
		}
		break;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Refine the search around the best points

bool GreedyExplorer::refineSearch()
{
	/*
	// Make a copy of the best patterns (they will be modified with each <searchAround> call)
	const int n_best_points = m_data->m_patternSpace.getNoBest();
	for (int i = 0; i < n_best_points; i ++)
	{
		m_best_patterns[i].copy(m_data->m_patternSpace.getBest(i));

	}

	// Search around each best pattern point
	for (int i = 0; i < n_best_points; i ++)
	{
		if (searchAround(m_best_patterns[i]) == false)
		{
			Torch::message("GreedyExplorer::refineSearch - error searching around some point!");
			return false;
		}
	}
	*/

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Check if the search should be stopped
//	(it is becoming too fine or no pattern found so far?!)

bool GreedyExplorer::shouldSearchMode(int old_n_candidates) const
{
	// If no pattern found so far, then the search should be stopped
	if (m_data->m_patterns.isEmpty() == true)
	{
		return false;
	}

	// If no pattern was added at the last iteration, ...
	if (m_data->m_patterns.size() == old_n_candidates)
	{
		return false;
	}

	// OK, keep on searching
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Profile some SW - fill the profile around it

bool GreedyExplorer::profileSW(int sw_x, int sw_y, int sw_w, int sw_h)
{
	const float dx = 0.01f * (VarX + 0.0f) * sw_w;
	const float dy = 0.01f * (VarY + 0.0f) * sw_h;

	// Vary the scale ...
	int index = 0;
	for (int is = -NoVarS; is <= NoVarS; is ++)
	{
		const float scale = 1.0f + 0.01f * (VarS + 0.0f) * is;
		const int new_sw_w = FixI(scale * sw_w);
		const int new_sw_h = FixI(scale * sw_h);

		// Vary the position ...
		for (int ix = -NoVarX; ix <= NoVarX; ix ++)
		{
			const int new_sw_x = sw_x + FixI(dx * ix);

			for (int iy = -NoVarY; iy <= NoVarY; iy ++)
			{
				const int new_sw_y = sw_y + FixI(dy * iy);

				// Default profile: no detection, low score
				m_profileFlags[index] = 0x00;
				m_profileScores[index] = -1000.0;

				// Process the sub-window, ignore if some error
				//      (the coordinates may fall out of the image)
				const int old_size = m_data->m_patterns.size();
				if (	ScaleExplorer::processSW(new_sw_x, new_sw_y, new_sw_w, new_sw_h, *m_data) &&
					m_data->m_patterns.size() != old_size)
				{
					m_profileScores[index] = m_data->m_patterns.get(old_size).m_confidence;
					m_profileFlags[index] = 0x01;
				}

				// Next profile
				index ++;
			}
		}
	}

	// OK
	return true;
}

bool GreedyExplorer::profileSW(const Pattern& pattern)
{
	return profileSW(pattern.m_x, pattern.m_y, pattern.m_w, pattern.m_h);
}

/////////////////////////////////////////////////////////////////////////

}
