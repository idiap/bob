#ifndef _TORCHVISION_SCANNING_SCALE_EXPLORER_H_
#define _TORCHVISION_SCANNING_SCALE_EXPLORER_H_

#include "Object.h"		// <ScaleExplorer> is a <Torch::Object>
#include "vision.h"		// <sSize> and <sRect2D> definitions
#include "Explorer.h"

namespace Torch
{
	class Tensor;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ScaleExplorer
	//	- scan some image (+ additional data, like integral image or edge maps)
	//			at a given scale
	//	- to evaluate any sub-window use the PUBLIC and STATIC functions:
	//		<initSW> and <processSW>,
	//		which will take care of prunning, evaluation and storing
	//		candidate sub-window patterns
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ScaleExplorer : public Torch::Object
	{
	public:

		// Constructor
		ScaleExplorer();

		// Destructor
		virtual ~ScaleExplorer();

		/////////////////////////////////////////////////////////////////
		// Process functions

		// Initialize the scanning process (scanning sub-window size, ROI)
		virtual bool		init(int sw_w, int sw_h, const sRect2D& roi);

		// Process the image (check for pattern's sub-windows)
		virtual bool		process(const Tensor& input_prune,
						const Tensor& input_evaluation,
						ExplorerData& explorerData,
						bool stopAtFirstDetection) = 0;

		/////////////////////////////////////////////////////////////////

		// Initialize the evaluator and pruners to some sub-window
		static void		initSW(	int sw_x, int sw_y, int sw_w, int sw_h,
						ExplorerData& explorerData);

		// Process some sub-window (already set for evaluator and pruners)
		static bool		processSW(	const Tensor& input_prune,
							const Tensor& input_evaluation,
							ExplorerData& explorerData);

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Scale to work with (scanning sub-window size)
		sSize			m_sw_size;

		// Current region of interest (ROI)
		// NB: can be different from the one in <Scanner> (check the pyramid approach!)
		sRect2D			m_roi;
	};
}

#endif
