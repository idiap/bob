#include "RandomScaleExplorer.h"
#include "ipSWEvaluator.h"

#include <cstdlib>

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

RandomScaleExplorer::RandomScaleExplorer()
	: 	ScaleExplorer()
{
	addIOption("NSamples", 128, "number of random points to generate");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

RandomScaleExplorer::~RandomScaleExplorer()
{
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process (scanning sub-window size, ROI)

bool RandomScaleExplorer::init(int sw_w, int sw_h, const sRect2D& roi)
{
	return ScaleExplorer::init(sw_w, sw_h, roi);
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool RandomScaleExplorer::process(	const Tensor& input_prune,
					const Tensor& input_evaluation,
					ExplorerData& explorerData,
					bool stopAtFirstDetection)
{
	const bool verbose = getBOption("verbose");

	///////////////////////////////////////////////////////////////////////
	// Generate all possible sub-windows to scan

	const int sw_w = m_sw_size.w;
	const int sw_h = m_sw_size.h;

	// ... compute the range in which the position can vary
	const int sw_min_x = m_roi.x;
	const int sw_max_x = m_roi.x + m_roi.w - sw_w;
	const int sw_min_y = m_roi.y;
	const int sw_max_y = m_roi.y + m_roi.h - sw_h;

	const int n_random_points = getInRange(getIOption("NSamples"), 1, 1024);
	srand((unsigned int)time(0));

	// ... and vary randomly the position
	int count = 0;
	for (int i = 0; i < n_random_points; i ++)
	{
		const int sw_x = sw_min_x + rand() % (sw_max_x - sw_min_x);
		const int sw_y = sw_min_y + rand() % (sw_max_y - sw_min_y);

		// Initialize the prunners and evaluator to this sub-window
		if (ScaleExplorer::initSW(sw_x, sw_y, sw_w, sw_h, explorerData) == false)
		{
		       Torch::message("RandomScaleExplorer::process \
					- could not initialize some sub-window!\n");
			return false;
		}

		// Process the sub-window
		if (ScaleExplorer::processSW(input_prune, input_evaluation, explorerData) == false)
		{
			Torch::message("RandomScaleExplorer::process \
					- could not process some sub-window!\n");
			return false;
		}

		// Stop at the first detection if asked
		if (stopAtFirstDetection && explorerData.m_patternSpace.isEmpty() == false)
		{
			break;
		}

		count ++;
	}

	// ... debug message
	if (verbose == true)
	{
		Torch::print("\t[RandomScaleExplorer]: [%d] SWs [%dx%d]: pruned = %d, scanned = %d, accepted = %d\n",
				count, sw_w, sw_h,
				explorerData.m_stat_prunned,
				explorerData.m_stat_scanned,
				explorerData.m_stat_accepted);
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
