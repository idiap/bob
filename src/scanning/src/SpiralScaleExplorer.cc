#include "scanning/SpiralScaleExplorer.h"
#include "scanning/ipSWEvaluator.h"

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
// Process the scale, searching for patterns at different sub-windows

bool SpiralScaleExplorer::process(	ExplorerData& explorerData,
					bool stopAtFirstDetection)
{
	const int sw_w = m_sw_size.w;
	const int sw_h = m_sw_size.h;

	// Compute the location variance (related to the subwindow size)
	//const int model_w = explorerData.m_swEvaluator->getModelWidth();
	//const int model_h = explorerData.m_swEvaluator->getModelHeight();

	const int dx = getInRange(FixI(getFOption("dx") * sw_w), 1, sw_w);
	const int dy = getInRange(FixI(getFOption("dy") * sw_h), 1, sw_h);

	const bool verbose = getBOption("verbose");

	///////////////////////////////////////////////////////////////////////
	// Generate all possible sub-windows to scan

	const int sw_center_x = m_roi.x + m_roi.w / 2;
	const int sw_center_y = m_roi.y + m_roi.h / 2;

	const int n_spirales = std::min(     (m_roi.w - sw_w) / dx,
                                        (m_roi.h - sw_h) / dy);

	// Vary the radius to the center of ROI ...
	int count = 0;
	for (int i = 0; i < n_spirales; i ++)
	{
	        // Compute the radius and the angle variation
	        const int radius_x = dx * i;
	        const int radius_y = dy * i;
	        const double theta = i == 0 ? 0.0 : (2.0 * asin(1.0 / (2.0 * i)));
	        const int n_thetas = i == 0 ? 1 : (int)(0.5 + (2.0 * M_PI) / theta);

	        //print ("[%d/%d]: radius_x = %d, radius_y = %d, theta = %f, n_thetas = %d\n",
                //      i + 1, n_spirales, radius_x, radius_y, theta, n_thetas);

                // Generate the sub-windows by varying the angle at the computed radius
                for (int j = 0; j < n_thetas; j ++)
                {
                        const double angle = theta * j;
                        const int center_x = sw_center_x + (int)(0.5 + cos(angle) * radius_x);
                        const int center_y = sw_center_y + (int)(0.5 + sin(angle) * radius_y);

                        const int sw_x = center_x - sw_w / 2;
                        const int sw_y = center_y - sw_h / 2;
                        if (sw_x < m_roi.x || sw_y < m_roi.y || sw_x + sw_w >= m_roi.w || sw_y + sw_h >= m_roi.h)
                        {
                                continue;
                        }

                        // Process the sub-window
                        if (ScaleExplorer::processSW(sw_x, sw_y, sw_w, sw_h, explorerData) == true)
                        {
				// Stop at the first detection if asked
				if (stopAtFirstDetection && explorerData.m_patterns.isEmpty() == false)
				{
					// This will exit gracefully the double <for>
					j = n_thetas;
					i = n_spirales;
				}

				count ++;
                        }
                }
	}

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
