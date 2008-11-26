#ifndef _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_
#define _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_

#include "MSExplorer.h"		// <GreedyExplorer> is a <MSExplorer>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::GreedyExplorer
	//	- searches the 4D scanning space using a greedy method to detect local maximas
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"Nbest"		int	128	"best N candidate patterns to consider/step"
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

		// Initialize the scanning 4D space (random or using a fixed grid ?!)
		bool				initSearch();

		// Refine the search around the best points
		bool				refineSearch();

		// Check if the search should be stopped
		//	(it is becoming too fine or no pattern found so far?!)
		bool				shouldSearchMode(int old_n_candidates = -1) const;

		// Search around some point in the position and scale space
		bool				searchAround(int x, int y, float scale);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Flag to check if more steps are needed
		bool				m_hasMoreSteps;

		// Current search parameters
		int				m_search_dx;
		int				m_search_dy;
		float				m_search_ds;

		// Minimum values for the search parameters (need to check if the search should be stopped)
		int				m_search_min_dx;
		int				m_search_min_dy;
		float				m_search_min_ds;

		// Keep a copy of the best patterns at some step
		Pattern*			m_best_patterns;
	};
}

#endif
