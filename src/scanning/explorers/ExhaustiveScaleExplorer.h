#ifndef _TORCHVISION_SCANNING_EXHAUSTIVE_SCALE_EXPLORER_H_
#define _TORCHVISION_SCANNING_EXHAUSTIVE_SCALE_EXPLORER_H_

#include "ScaleExplorer.h"	// <ExhaustiveScaleExplorer> is a <ScaleExplorer>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::ExhaustiveScaleExplorer
	//	- investigates all possible sub-windows at a specified scale
	//		in a 2D grid fashion, given some variance on the position
	//		(dx and dy parameters)
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"dx"	float	0.1	"OX variation of the pattern width"
	//		"dy"	float	0.1	"OY variation of the pattern height"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ExhaustiveScaleExplorer : public ScaleExplorer
	{
	public:

		// Constructor
		ExhaustiveScaleExplorer();

		// Destructor
		virtual ~ExhaustiveScaleExplorer();

		/////////////////////////////////////////////////////////////////
		// Process functions

		// Initialize the scanning process (scanning sub-window size, ROI)
		virtual bool		init(int sw_w, int sw_h, const sRect2D& roi);

		// Process the image (check for pattern's sub-windows)
		virtual bool		process(const Tensor& input_prune,
						const Tensor& input_evaluation,
						ExplorerData& explorerData,
						bool stopAtFirstDetection);

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
