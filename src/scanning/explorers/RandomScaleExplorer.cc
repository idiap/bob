#include "RandomScaleExplorer.h"
#include "ipSWEvaluator.h"

#include <stdlib.h>

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
// Process the scale, searching for patterns at different sub-windows

bool RandomScaleExplorer::process(	ExplorerData& explorerData,
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

		// Process the sub-window
		if (ScaleExplorer::processSW(sw_x, sw_y, sw_w, sw_h, explorerData) == true)
		{
			// Stop at the first detection if asked
			if (stopAtFirstDetection && explorerData.m_patterns.isEmpty() == false)
			{
				break;
			}

			count ++;
		}
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
