#include "GreedyExplorer.h"
#include "ipSWEvaluator.h"
#include "ScaleExplorer.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

GreedyExplorer::GreedyExplorer(ipSWEvaluator* swEvaluator)
	: 	MSExplorer(swEvaluator),
		m_search_per_dx(0.1f),
		m_search_per_dy(0.1f),
		m_search_per_ds(0.1f),
		m_search_no_steps(5),
		m_best_patterns(0)
{
	addIOption("Nbest", 128, "best N candidate patterns to consider/step");
	addIOption("SWdx", 10, "% of the sub-window width to vary Ox when refining the search");
	addIOption("SWdy", 10, "% of the sub-window height to vary Oy when refining the search");
	addIOption("SWds", 10, "% of the sub-window size to vary scale when refining the search");
	addIOption("NoSteps", 5, "number of iterations");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

GreedyExplorer::~GreedyExplorer()
{
	delete[] m_best_patterns;
}

/////////////////////////////////////////////////////////////////////////
// called when some option was changed - overriden

void GreedyExplorer::optionChanged(const char* name)
{
       m_search_per_dx = 0.01f * getInRange(getIOption("SWdx"), 1, 100);
       m_search_per_dy = 0.01f * getInRange(getIOption("SWdy"), 1, 100);
       m_search_per_ds = 0.01f * getInRange(getIOption("SWds"), 1, 100);
       m_search_no_steps = getInRange(getIOption("NoSteps"), 1, 100);
}

/////////////////////////////////////////////////////////////////////////
// Check if the scanning can continue (or the space was explored enough)

bool GreedyExplorer::hasMoreSteps() const
{
	return m_stepCounter < m_search_no_steps;
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool GreedyExplorer::process()
{
	// Already finished!
	if (m_stepCounter >= m_search_no_steps)
	{
		Torch::message("GreedyExplorer::process - the processing already finished!");
		return false;
	}

	const bool verbose = getBOption("verbose");

	const int old_n_candidates = m_data->m_patternSpace.size();

	// First iteration - need to initialize the scanning 4D space
	if (m_stepCounter == 0)
	{
		// Don't want to stop at the first iteration or to scale from large to small windows
		setBOption("StopAtFirstDetection", false);
		setBOption("StartWithLargeScales", false);

		// Allocate the best patterns (just a copy from the one in m_patternSpace), if it's needed
		const int n_best = getInRange(getIOption("Nbest"), 1, 1024);
		if (	n_best != m_data->m_patternSpace.getMaxNoBest() ||
			m_best_patterns == 0)
		{
			delete[] m_best_patterns;
			m_best_patterns = new Pattern[n_best];

			// Make sure the pattern space is resized in respect to the number of best patterns too!
			m_data->m_patternSpace.reset(n_best);
		}

                // Initialize the scanning 4D space
		if (initSearch() == false)
		{
			Torch::message("GreedyExplorer::process - error initializing search space!\n");
			return false;
		}
	}

	// Refine the search around the best points
	else
	{
                if (refineSearch() == false)
                {
                        Torch::message("GreedyExplorer::process - error refining the search space!\n");
                        return false;
                }
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
	        m_stepCounter = m_search_no_steps;

	        // Debug message
                if (verbose == true)
                {
                        Torch::print("[GreedyExplorer]: stopping ...\n");
                }
	}

	// OK, next step
	m_stepCounter ++;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning 4D space (random or using a fixed grid ?!)

bool GreedyExplorer::initSearch()
{
	// Just call the MSExplorer (which will run <ScaleExplorer>s at each scale)
	return MSExplorer::process();
}

/////////////////////////////////////////////////////////////////////////
// Refine the search around the best points

bool GreedyExplorer::refineSearch()
{
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

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Check if the search should be stopped
//	(it is becoming too fine or no pattern found so far?!)

bool GreedyExplorer::shouldSearchMode(int old_n_candidates) const
{
	// If no pattern found so far, then the search should be stopped
	if (m_data->m_patternSpace.isEmpty() == true)
	{
		return false;
	}

	// If no pattern was added at the last iteration, ...
	if (m_data->m_patternSpace.size() == old_n_candidates)
	{
		return false;
	}

	// OK, keep on searching
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Search around some point in the position and scale space (= sub-window)

bool GreedyExplorer::searchAround(const Pattern& candidate)
{
        static const int n_scales = 5;
	static const int n_positions = 5;

        const int model_w = getModelWidth();
	const int model_h = getModelHeight();

	// Compute the variations in space and scale
	const int sw_x_delta = (int)(0.5f * m_search_per_dx * candidate.m_w);
        const int min_sw_x = candidate.m_x - sw_x_delta;
        const int max_sw_x = candidate.m_x + sw_x_delta;
        const float delta_x = (max_sw_x - min_sw_x + 0.0f) / (n_positions - 1.0f);

        const int sw_y_delta = (int)(0.5f + m_search_per_dy * candidate.m_h);
        const int min_sw_y = candidate.m_y - sw_y_delta;
        const int max_sw_y = candidate.m_y + sw_y_delta;
        const float delta_y = (max_sw_y - min_sw_y + 0.0f) / (n_positions - 1.0f);

        const float scale = (candidate.m_w + 0.0f) / (model_w + 0.0f);
        const float min_scale = scale * (1.0f - m_search_per_ds);
        const float max_scale = scale * (1.0f + m_search_per_ds);
        const float delta_scale = (max_scale - min_scale) / (n_scales - 1.0f);

        // Vary the scale ...
	for (int is = 0; is < n_scales; is ++)
	{
		const float new_scale = min_scale + delta_scale * is;
		const int sw_w = (int)(0.5f + new_scale * model_w);
		const int sw_h = (int)(0.5f + new_scale * model_h);

		// Vary the position ...
		for (int ix = 0; ix < n_positions; ix ++)
			for (int iy = 0; iy < n_positions; iy ++)
			{
				const int sw_x = min_sw_x + (int)(0.5f + delta_x * ix);
				const int sw_y = min_sw_y + (int)(0.5f + delta_y * iy);

				// Skip this subwindow if it was scanned before
				if (m_data->m_patternSpace.hasPoint(sw_x, sw_y, sw_w, sw_h) == true)
				{
				        continue;
				}

				// Process the sub-window, ignore if some error
				//      (the coordinates may fall out of the image)
				if (ScaleExplorer::processSW(sw_x, sw_y, sw_w, sw_h, *m_data) == false)
				{
                                        //continue;
				}
			}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
