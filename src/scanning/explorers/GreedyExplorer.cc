#include "GreedyExplorer.h"
#include "ipSWDummyEvaluator.h"
#include "ScaleExplorer.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

GreedyExplorer::GreedyExplorer(ipSWEvaluator* swEvaluator)
	: 	MSExplorer(swEvaluator),
		m_hasMoreSteps(false),
		m_search_dx(1), m_search_dy(1), m_search_ds(0.5f),
		m_search_min_dx(1), m_search_min_dy(1), m_search_min_ds(0.01f),
		m_best_patterns(0)
{
	addIOption("Nbest", 128, "best N candidate patterns to consider/step");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

GreedyExplorer::~GreedyExplorer()
{
	delete[] m_best_patterns;
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process with the given image size

bool GreedyExplorer::init(int image_w, int image_h)
{
	m_hasMoreSteps = true;
	return 	MSExplorer::init(image_w, image_h);
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process for a specific ROI

bool GreedyExplorer::init(const sRect2D& roi)
{
	m_hasMoreSteps = true;
	return MSExplorer::init(roi);
}

/////////////////////////////////////////////////////////////////////////
// Check if the scanning can continue (or the space was explored enough)

bool GreedyExplorer::hasMoreSteps() const
{
	return m_hasMoreSteps;
}

/////////////////////////////////////////////////////////////////////////
// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>

bool GreedyExplorer::preprocess(const Image& image)
{
	// It's enough the preprocessing from the MSExplorer
	return MSExplorer::preprocess(image);
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool GreedyExplorer::process()
{
	// Already finished!
	if (m_hasMoreSteps == false)
	{
		Torch::message("GreedyExplorer::process - the processing already finished!");
		return false;
	}

	const bool verbose = getBOption("verbose");

	// First iteration - need to initialize the scanning 4D space
	if (m_stepCounter == 0)
	{
		// Don't want to stop at the first iteration or to scale from large to small windows
		setBOption("StopAtFirstDetection", false);
		setBOption("StartWithLargeScales", false);

		const int model_w = getModelWidth();
		const int model_h = getModelHeight();

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

		// Initialize the searching parameters for each axis
		m_search_ds = m_n_scales <= 1 ? 0.5f : (m_scales[1].w - m_scales[0].w + 0.0f) / (model_w + 0.0f);
		m_search_dx = model_w / 2;
		m_search_dy = model_h / 2;

		// Compute the minimum values for the searching parameters
		//	... at least one pixel in location variation and
		m_search_min_ds = 1.0f / (min(model_w, model_h) + 0.0f);
		m_search_min_dx = 1;
		m_search_min_dy = 1;

		// Debug message
		if (verbose == true)
		{
			Torch::print("[GreedyExplorer]: searching parameters - dx = %d, dy = %d, ds = %f\n",
					m_search_dx, m_search_dy, m_search_ds);
			Torch::print("[GreedyExplorer]: initializing the pattern space ...\n");
		}

		// Initialize the scanning 4D space (random or using a fixed grid ?!)
		if (initSearch() == false)
		{
			Torch::message("GreedyExplorer::process - error initializing search space!\n");
			return false;
		}

		// Debug message
		if (verbose == true)
		{
			Torch::print("[GreedyExplorer]: initialized, pruned = %d, scanned = %d, accepted = %d\n",
					m_data->m_stat_prunned,
					m_data->m_stat_scanned,
					m_data->m_stat_accepted);
		}

		// Check if the search should be stopped
		//	(no pattern found so far?!)
		m_hasMoreSteps = shouldSearchMode();
	}

	else
	{
		// Check if the search should be stopped
		//	(it is becoming too fine or no pattern found so far?!)
		m_hasMoreSteps = shouldSearchMode();

		// ... more search steps are needed
		if (m_hasMoreSteps == true)
		{
			// Refine the searching parameters for each axis (if it's possible)
			m_search_ds = min(m_search_ds * 0.5f, m_search_min_ds);
			m_search_dx = min(m_search_dx / 2, m_search_min_dx);
			m_search_dy = min(m_search_dy / 2, m_search_min_dy);

			// Debug message
			if (verbose == true)
			{
				Torch::print("[GreedyExplorer]: searching parameters: dx = %d, dy = %d, ds = %f\n",
					m_search_dx, m_search_dy, m_search_ds);
				Torch::print("[GreedyExplorer]: refining the search space ...\n");
			}

			// Refine the search around the best points
			const int old_n_candidates = m_data->m_patternSpace.size();
			if (refineSearch() == false)
			{
				Torch::message("GreedyExplorer::process - error refining the search space!\n");
				return false;
			}

			// Debug message
			if (verbose == true)
			{
				Torch::print("[GreedyExplorer]: refined, pruned = %d, scanned = %d, accepted = %d\n",
						m_data->m_stat_prunned,
						m_data->m_stat_scanned,
						m_data->m_stat_accepted);
			}

			// Check the stopping criterion one more time
			//	(maybe the search parameters became too small or no other pattern was added)
			m_hasMoreSteps = shouldSearchMode(old_n_candidates);
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
	const float inv_model_w = 1.0f / (m_data->m_swEvaluator->getModelWidth());
	for (int i = 0; i < n_best_points; i ++)
	{
		const Pattern& pattern = m_best_patterns[i];
		const float scale = inv_model_w * pattern.m_w;

		if (searchAround(pattern.m_x, pattern.m_y, scale) == false)
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

	// If search parameters are ALL to small, ...
	if (	m_search_dx == m_search_min_dx &&
		m_search_dy == m_search_min_dy &&
		fabs(m_search_ds - m_search_min_ds) < 0.001f)
	{
		return false;
	}

	// OK, keep on searching
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Search around some point in the position and scale space

bool GreedyExplorer::searchAround(int x, int y, float scale)
{
	const int model_w = getModelWidth();
	const int model_h = getModelHeight();

	// Dinamic scales
	//	=> the pre-processing algorithm may not know about these scale
	//	=> we'll use the -1 index, to get the tensor data to be processed by the prunners and evaluator
	//	=> the pre-processing algorithm should take care of this
	const int scale_index = -1;

	// Vary the scale (+/- variance) ...
	for (int is = 0; is < 3; is ++)
	{
		const float new_scale = scale + (is - 1.0f) * m_search_ds;
		const int sw_w = (int)(0.5f + new_scale * model_w);
		const int sw_h = (int)(0.5f + new_scale * model_h);

		// Vary the position (+/- variance) ...
		for (int ix = 0; ix < 3; ix ++)
			for (int iy = 0; iy < 3; iy ++)
				if (is != 1 || ix != 1 || iy != 1)	// To avoid the current point to be scanned again
			{
				const int sw_x = x + (ix - 1) * m_search_dx;
				const int sw_y = y + (iy - 1) * m_search_dy;

				// Process the computed sub-window
				ScaleExplorer::initSW(sw_x, sw_y, sw_w, sw_h, *m_data);
				if (ScaleExplorer::processSW(	m_scale_prune_ips[0]->getOutput(0),
								m_scale_evaluation_ips[0]->getOutput(0),
								*m_data) == false)
				{
					Torch::message("GreedyExplorer::searchAround -\
						error processing some sub-window!\n");
					return false;
				}
			}
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
