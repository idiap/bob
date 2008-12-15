#ifndef _TORCHVISION_SCANNING_EXPLORER_H_
#define _TORCHVISION_SCANNING_EXPLORER_H_

#include "Object.h"		// <Explorer> is a <Torch::Object>
#include "Pattern.h"		// works on <Pattern>s
#include "vision.h"		// <sSize> and <sRect2D> definitions

namespace Torch
{
	class Image;
	class ipSWPruner;
	class ipSWEvaluator;
	class ipCore;
	class ScaleExplorer;
	class Tensor;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ExplorerData:
	//	- groups the objects needed for the scanning:
	//		- 1 x evaluator
	//		- N x pruners
	//		- 1 x pattern space
	//		- additional statistical information
	//	- used for passing this information to the <ScaleExplorer>s
	//	- store some pattern
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct ExplorerData
	{
		// Constructor
		ExplorerData(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~ExplorerData();

		// Initialize the (evaluator + pruners) processing for these tensors
		bool                    init(   const Tensor& input_prune,
                                                const Tensor& input_evaluation);

		// Delete all the detections so far
		void			clear();

		// Store some pattern (may need rescalling the sub-window!)
		// NB: the MS approach should just copy it, the pyramid should rescalling it
		//	before storing the PatternSpace
		virtual void		storePattern(	int sw_x, int sw_y, int sw_w, int sw_h,
							float confidence) = 0;

		//////////////////////////////////////////////////////////////////
		// Attributes

		// Object for checking if some sub-window contains the pattern
		ipSWEvaluator*		m_swEvaluator;

		// Objects to prune some sub-windows before the model is run on them
		ipSWPruner**		m_swPruners;
		int			m_nSWPruners;

		// Image size to scan
		int			m_image_w;
		int			m_image_h;

		// Statistical information
		int			m_stat_scanned;	// Number of scanned sub-windows by the pattern model
		int			m_stat_prunned;	// Number of prunned sub-windows
		int			m_stat_accepted;// Number of sub-windows with patterns
			// Total number of investigated sub-windows = m_stat_scanned + m_stat_prunned
			// m_stat_scanned >= m_stat_accepted

		PatternSpace		m_patternSpace;
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::Explorer
	//	- explore the 4D (position + scale + model confidence) scanning space!
	//	- decide what sub-windows to scan the next step
	//	- decide when to stop
	//      - if the <ipCore>s for prunning and evaluation at scales is NULL then
	//              the original (scaled) image will be used
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"min_patt_w"		integer		0	"pattern min allowed width"
	//		"max_patt_w"		integer		4096	"pattern max allowed width"
	//		"min_patt_h"		integer		0	"pattern min allowed height"
	//		"max_patt_h"		integer		4096	"pattern max allowed height"
	//		"ds"			float		1.25	"scale variation from the smallest to the largest window size"
        //                            (1 < ds < inf BUT IN REALITY (max_window_size / min_window_size))
	//		"StopAtFirstDetection"	bool		false	"stop at the first candidate patterns"
	//		"StartWithLargeScales"	bool		false	"large to small scales scanning"
	//
	//	- !!!should be used inside a Scanner object, which better enforce conditions
	//		and checks for possible errors (null pointers, invalid values etc.)!!!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class Explorer : public Torch::Object
	{
	public:
		/////////////////////////////////////////////////////////////////

		// Constructor
		Explorer(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~Explorer();

		/////////////////////////////////////////////////////////////////
		// Modify the scanning objects

		// Modify the sub-window pattern model operator
		bool			setSWEvaluator(ipSWEvaluator* swEvaluator);

		// Modify the sub-window prunning operators
		bool			addSWPruner(ipSWPruner* swPruner);
		void			deleteAllSWPruners();

		/////////////////////////////////////////////////////////////////

		// Set the scanning strategies for the scales (different or the same)
		bool			setScaleExplorer(ScaleExplorer* scaleExplorer);
		bool			setScaleExplorer(int index_scale, ScaleExplorer* scaleExplorer);

		// Set the features to use for the scales (different or the same)
		//	(for prunning and pattern evaluation)
		// (If they are 0/NULL, then the original input tensor will be used!)
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
		virtual bool		preprocess(const Image& image) = 0;

		// Process the image (check for pattern's sub-windows)
		virtual bool		process() = 0;

		/////////////////////////////////////////////////////////////////
		// Access functions

		int			getNoScannedSWs() const { return m_data->m_stat_scanned; }
		int			getNoPrunnedSWs() const { return m_data->m_stat_prunned; }
		int			getNoAcceptedSWs() const { return m_data->m_stat_accepted; }
		const PatternSpace&	getPatternSpace() const { return m_data->m_patternSpace; }
		int			getModelWidth() const;
		int			getModelHeight() const;
		int			getNoScales() const { return m_n_scales; }
		const sSize&		getScale(int index_scale) const;

		/////////////////////////////////////////////////////////////////

	protected:

		// Deallocate the scale information
		void			deallocateScales();

		// Resize the scale information
		bool			resizeScales(int n_scales);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Objects needed for checking a sub-window and storing results
		ExplorerData*		m_data;

		// The scales where to scan and for each scale:
		//	- the associated <ScaleExplorer>s
		//	- <ipCore>s for prunning and evaluation
		//	- ROIs
		sSize*			m_scales;
		ScaleExplorer**		m_scale_explorers;
		ipCore**		m_scale_prune_ips;
		ipCore**		m_scale_evaluation_ips;
		sRect2D*		m_scale_rois;
		int			m_n_scales;

		// Keep track of how many steps to get the result
		int			m_stepCounter;
	};
}

#endif
