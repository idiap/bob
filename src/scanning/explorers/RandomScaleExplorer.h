#ifndef _TORCHVISION_SCANNING_RANDOM_SCALE_EXPLORER_H_
#define _TORCHVISION_SCANNING_RANDOM_SCALE_EXPLORER_H_

#include "ScaleExplorer.h"	// <RandomScaleExplorer> is a <ScaleExplorer>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::RandomScaleExplorer
	//	- samples some random points at the specified scale
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"NSamples"	int	128		"number of random points to generate"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class RandomScaleExplorer : public ScaleExplorer
	{
	public:

		// Constructor
		RandomScaleExplorer();

		// Destructor
		virtual ~RandomScaleExplorer();

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
