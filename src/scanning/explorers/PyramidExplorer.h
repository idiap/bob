#ifndef _TORCHVISION_SCANNING_PYRAMID_EXPLORER_H_
#define _TORCHVISION_SCANNING_PYRAMID_EXPLORER_H_

#include "Explorer.h"		// <PyramidExplorer> is an <Explorer>

namespace Torch
{
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
							float confidence);

		// Set the current scanning scale
		void			setScale(const sSize& scale);

		///////////////////////////////////////////////////////////////
		// Attributes

		// Current scaled image size
		sSize			m_scale;
		float			m_inv_scale_w;
		float			m_inv_scale_h;
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::PyramidExplorer
	//	- builds a pyramid of image at different scales, while keeping the model
	//		at a fixed size
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class PyramidExplorer : public Explorer
	{
	public:

		// Constructor
		PyramidExplorer(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~PyramidExplorer();

		/////////////////////////////////////////////////////////////////

		// Set the features to use for the scales
		//	(for prunning and pattern evaluation)
		// It's enforced to use one pruneIp and evaluationIp per scale!
		//	=> the functions without <index_scale> will return false!!!

		virtual bool		setScalePruneIp(ipCore* scalePruneIp);
		virtual bool		setScalePruneIp(int index_scale, ipCore* scalePruneIp);
		virtual bool		setScaleEvaluationIp(ipCore* scaleEvaluationIp);
		virtual bool		setScaleEvaluationIp(int index_scale, ipCore* scaleEvaluationIp);

		/////////////////////////////////////////////////////////////////
		// Process functions

		// HOW TO USE (see Scanner):
		// --------------------------------
		// init(image_w, image_h)
		// ... setScaleXXX as wanted
		// preprocess(image)
		// for each ROI
		//	init (ROI)
		// 	while (hasMoreSteps())
		//		process ()
		// --------------------------------

		// Initialize the scanning process with the given image size
		virtual bool		init(int image_w, int image_h);

		// Initialize the scanning process for a specific ROI
		virtual bool		init(const sRect2D& roi);

		// Check if the scanning can continue (or the space was explored enough)
		virtual bool		hasMoreSteps() const;

		// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>
		virtual bool		preprocess(const Image& image);

		// Process the image (check for pattern's sub-windows)
		virtual bool		process();

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
