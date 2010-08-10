#ifndef _TORCHVISION_SCANNING_MS_EXPLORER_H_
#define _TORCHVISION_SCANNING_MS_EXPLORER_H_

#include "scanning/Explorer.h"		// <MSExplorer> is an <Explorer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::MSExplorerData:
	//	- implementation of <ExplorerData>, just modifying the way
	//		candidate patterns are stored
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct MSExplorerData : public ExplorerData
	{
		// Constructor
		MSExplorerData(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~MSExplorerData();

		// Store some pattern - just copy it!
		virtual void		storePattern(	int sw_x, int sw_y, int sw_w, int sw_h,
							double confidence);
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::MSExplorer
	//	- MultiScale explorer
	//	- keep the image size and vary the size of model/template (window scan)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class MSExplorer : public Explorer
	{
	public:
		// Constructor
		MSExplorer();

		// Destructor
		virtual ~MSExplorer();

		/////////////////////////////////////////////////////////////////
		// Process functions

		// HOW TO USE (see Scanner):
		// --------------------------------
		// init(image_w, image_h)
		// ... setScaleXXX as wanted
		// preprocess(image)
		// for each ROI
		//	init (ROI)
		// 	process ()
		// --------------------------------

		// Initialize the scanning process with the given image size
		virtual bool		init(int image_w, int image_h);

		// Initialize the scanning process for a specific ROI
		virtual bool		init(const sRect2D& roi);

		// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>
		virtual bool		preprocess(const Image& image);

		// Process the image (check for pattern's sub-windows)
		virtual bool		process();

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Tensors (features) for the pruning and evaluation
		const Tensor*           m_prune_tensor;
		const Tensor*           m_evaluation_tensor;
	};
}

#endif
