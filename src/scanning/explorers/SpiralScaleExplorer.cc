#include "SpiralScaleExplorer.h"
#include "ipSWEvaluator.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

SpiralScaleExplorer::SpiralScaleExplorer()
	: 	ScaleExplorer()
{
	addFOption("dx", 0.1f, "OX variation of the pattern width");
	addFOption("dy", 0.1f, "OY variation of the pattern height");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

SpiralScaleExplorer::~SpiralScaleExplorer()
{
}

/////////////////////////////////////////////////////////////////////////
// Initialize the scanning process (scanning sub-window size, ROI)

bool SpiralScaleExplorer::init(int sw_w, int sw_h, const sRect2D& roi)
{
	return ScaleExplorer::init(sw_w, sw_h, roi);
}

/////////////////////////////////////////////////////////////////////////
// Process the image (check for pattern's sub-windows)

bool SpiralScaleExplorer::process(	const Tensor& input_prune,
					const Tensor& input_evaluation,
					ExplorerData& explorerData,
					bool stopAtFirstDetection)
{
	// Compute the location variance (related to the model size)
	const int model_w = explorerData.m_swEvaluator->getModelWidth();
	const int model_h = explorerData.m_swEvaluator->getModelHeight();

	const int dx = getInRange((int)(0.5f + getFOption("dx") * model_w), 1, model_w);
	const int dy = getInRange((int)(0.5f + getFOption("dy") * model_h), 1, model_h);

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

	// ... and vary the position
	int count = 0;
	/* TODO
	for (int sw_x = sw_min_x; sw_x <= sw_max_x; sw_x += dx)
		for (int sw_y = sw_min_y; sw_y <= sw_max_y; sw_y += dy)
		{
			// Initialize the prunners and evaluator to this sub-window
			if (ScaleExplorer::initSW(sw_x, sw_y, sw_w, sw_h, explorerData) == false)
			{
                                Torch::message("SpiralScaleExplorer::process \
						- could not initialize some sub-window!\n");
				return false;
			}

			// Process the sub-window
			if (ScaleExplorer::processSW(input_prune, input_evaluation, explorerData) == false)
			{
				Torch::message("SpiralScaleExplorer::process \
						- could not process some sub-window!\n");
				return false;
			}

			// Stop at the first detection if asked
			if (stopAtFirstDetection && explorerData.m_patternSpace.isEmpty() == false)
			{
				// This will exit gracelly the double <for>
				sw_x = sw_max_x + 1;
				sw_y = sw_max_y + 1;
			}

			count ++;
		}
	*/

	// ... debug message
	if (verbose == true)
	{
		Torch::print("\t[SpiralScaleExplorer]: [%d] SWs [%dx%d]: pruned = %d, scanned = %d, accepted = %d\n",
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
