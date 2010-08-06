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
	//		"dx"	float	0.1	"OX variation of the scanning sub-window width"
	//		"dy"	float	0.1	"OY variation of the scanning sub-window height"
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

		// Process the scale, searching for patterns at different sub-windows
		virtual bool		process(ExplorerData& explorerData,
						bool stopAtFirstDetection);

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
