#ifndef _TORCHVISION_SCANNING_SCALE_EXPLORER_H_
#define _TORCHVISION_SCANNING_SCALE_EXPLORER_H_

#include "core/Object.h"		// <ScaleExplorer> is a <Torch::Object>
#include "ip/vision.h"		// <sSize> and <sRect2D> definitions
#include "Explorer.h"

namespace Torch
{
	class Tensor;

	/////////////////////////////////////////////////////////////////////////
	// Torch::ScaleExplorer
	//	- scan some image (+ additional data, like integral image or edge maps)
	//			at a given scale
	//	- to evaluate any sub-window use the PUBLIC and STATIC function <processSW>,
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
		bool	        	init(int sw_w, int sw_h, const sRect2D& roi);

		// Process the scale, searching for patterns at different sub-windows
		virtual bool		process(ExplorerData& explorerData,
						bool stopAtFirstDetection) = 0;

		// Process some sub-window
		static bool		processSW(      int sw_x, int sw_y, int sw_w, int sw_h,
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
