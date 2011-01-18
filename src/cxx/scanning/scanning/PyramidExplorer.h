#ifndef _TORCHVISION_SCANNING_PYRAMID_EXPLORER_H_
#define _TORCHVISION_SCANNING_PYRAMID_EXPLORER_H_

#include "scanning/Explorer.h"		// <PyramidExplorer> is an <Explorer>

namespace Torch
{
        class ipScaleYX;
        class Tensor;
        class Image;

	/////////////////////////////////////////////////////////////////////////
	// Torch::PyramidExplorerData:
	//	- implementation of <ExplorerData>, just modifying the way
	//		candidate patterns are stored
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct PyramidExplorerData : public ExplorerData
	{
		// Constructor
		PyramidExplorerData(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~PyramidExplorerData();

		// Store some pattern - just copy it!
		virtual void		storePattern(	int sw_x, int sw_y, int sw_w, int sw_h,
							double confidence);

		// Set the current scanning scale
		void			setScale(const sSize& scale);

		///////////////////////////////////////////////////////////////
		// Attributes

		// Current scaled image size
		sSize			m_scale;
		float			m_inv_scale;
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::PyramidExplorer
	//	- builds a pyramid of image at different scales, while keeping the model
	//		at a fixed size
	//
	//      - PARAMETERS (name, type, default value, description):
	//		"savePyramidsToJpg"	bool    false   "save the scaled images to JPEG"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class PyramidExplorer : public Explorer
	{
	public:

		// Constructor
		PyramidExplorer();

		// Destructor
		virtual ~PyramidExplorer();

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

		const Image*		m_image;
	};
}

#endif
