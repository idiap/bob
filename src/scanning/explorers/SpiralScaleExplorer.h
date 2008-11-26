#ifndef _TORCHVISION_SCANNING_SPIRAL_SCALE_EXPLORER_H_
#define _TORCHVISION_SCANNING_SPIRAL_SCALE_EXPLORER_H_

#include "ScaleExplorer.h"	// <SpiralScaleExplorer> is a <ScaleExplorer>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::SpiralScaleExplorer
	//	- investigates all possible sub-windows at a specified scale
	//		in spiral order from the center of the image,
	//		given some variance on the position (dx and dy parameters)
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"dx"	float	0.1	"OX variation of the pattern width"
	//		"dy"	float	0.1	"OY variation of the pattern height"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class SpiralScaleExplorer : public ScaleExplorer
	{
	public:

		// Constructor
		SpiralScaleExplorer();

		// Destructor
		virtual ~SpiralScaleExplorer();

		/////////////////////////////////////////////////////////////////
		// Process functions

		// Initialize the scanning process (scanning sub-window size, ROI)
		virtual bool		init(int sw_w, int sw_h, const sRect2D& roi);

		// Process the image (check for pattern's sub-windows)
		virtual bool		process(const Tensor& input_prune,
						const Tensor& input_evaluation,
						ExplorerData& explorerData,
						bool stopAtFirstDetecton);

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
