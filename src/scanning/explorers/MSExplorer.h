#ifndef _TORCHVISION_SCANNING_MS_EXPLORER_H_
#define _TORCHVISION_SCANNING_MS_EXPLORER_H_

#include "Explorer.h"		// <MSExplorer> is an <Explorer>
#include "ipIntegral.h"         // Uses <ipIntegral> for fast scalling

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
							float confidence);
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
		MSExplorer(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~MSExplorer();

		/////////////////////////////////////////////////////////////////

		// Set the features to use for the scales
		//	(for prunning and pattern evaluation)
		// It's enforced to use the same pruneIp and evaluationIp!
		//	=> the <index_scale> functions will return false!!!

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

                // <ipIntegral> to compute the integral image for the prune/evaluation tensors
                // (Needed to fast scale the scanning sub-window to the model size)
		ipIntegral              m_ipi_prune;
		ipIntegral              m_ipi_evaluation;

		// Integral tensors (features) for the pruning and evaluation
		// (for the whole image, pointing to <ipIntegral>s above)
		const Tensor*           m_prune_itensor;
		const Tensor*           m_evaluation_itensor;
	};
}

#endif
