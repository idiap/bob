#ifndef _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_
#define _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_

#include "MSExplorer.h"		// <GreedyExplorer> is a <MSExplorer>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::GreedyExplorer
	//	- searches the 4D scanning space using a greedy method to refine
	//              the search around the best sub-windows
	//              (relative to the model confidence)
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"Nbest"		int	128	"best N candidate patterns to consider/step"
	//              "SWdx"          int     10      "% of the sub-window width to vary Ox when refining the search"
	//              "SWdy"          int     10      "% of the sub-window height to vary Oy when refining the search"
	//              "SWds"          int     10      "% of the sub-window size to vary scale when refining the search"
	//              "NoSteps"       int     5       "number of iterations"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class GreedyExplorer : public MSExplorer
	{
	public:

		// Constructor
		GreedyExplorer(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~GreedyExplorer();

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

		// Check if the scanning can continue (or the space was explored enough)
		virtual bool		hasMoreSteps() const;

		// Process the image (check for pattern's sub-windows)
		virtual bool		process();

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////

		/// called when some option was changed - overriden
		virtual void		optionChanged(const char* name);

		// Initialize the scanning 4D space (random or using a fixed grid ?!)
		bool			initSearch();

		// Refine the search around the best points
		bool			refineSearch();

		// Check if the search should be stopped
		//	(it is becoming too fine or no pattern found so far?!)
		bool			shouldSearchMode(int old_n_candidates = -1) const;

		// Search around some point in the position and scale space (= sub-window)
		bool			searchAround(const Pattern& candidate);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Current search parameters
		float                   m_search_per_dx;
		float                   m_search_per_dy;
		float                   m_search_per_ds;
		int                     m_search_no_steps;

		// Keep a copy of the best patterns at some step
		Pattern*		m_best_patterns;
	};
}

#endif
