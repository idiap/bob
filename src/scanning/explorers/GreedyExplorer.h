#ifndef _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_
#define _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_

#include "MSExplorer.h"		// <GreedyExplorer> is a <MSExplorer>
#include "MeanShiftSelector.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::GreedyExplorer
	//	- searches the 4D scanning space using a greedy method to refine
	//              the search around the best sub-windows
	//              (relative to the model confidence)
	//
	//	- PARAMETERS (name, type, default value, description):
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class GreedyExplorer : public MSExplorer
	{
	public:

		// Running mode
		enum Mode
		{
			Scanning,	// Iterative & greedy scanning (using the trained model for FAs)
			Profiling	// Generates profiles and saves them to output file
		};

		// Profiling constants
		static const int	NoVarX = 6;	// No. of steps on Ox
		static const int	NoVarY = 6;	// No. of steps on Oy
		static const int 	NoVarS = 7;	// No. of steps on scale
		static const int	VarX = 5;	// %/step variation on Ox
		static const int	VarY = 5;	// %/step variation on Oy
		static const int	VarS = 5;	// %/step variation on scales
		static const int	NoConfigs = 	(2 * NoVarX + 1) *
							(2 * NoVarY + 1) *
							(2 * NoVarS + 1);

		// Constructor
		GreedyExplorer(ipSWEvaluator* swEvaluator = 0, Mode = Scanning);

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
		// 	process ()
		// --------------------------------

		// Process the image (check for pattern's sub-windows)
		virtual bool		process();

		// Profile some SW - fill the profile around it
		bool			profileSW(int sw_x, int sw_y, int sw_w, int sw_h);
		bool			profileSW(const Pattern& pattern);

		// Access functions
		void			setMode(Mode mode) { m_mode = mode; }
		Mode 			getMode() const { return m_mode; }
		const unsigned char*	getProfileFlags() const { return m_profileFlags; }
		const double*		getProfileScores() const { return m_profileScores; }

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////

		/// called when some option was changed - overriden
		virtual void		optionChanged(const char* name);

		// Refine the search around the best points
		bool			refineSearch();

		// Check if the search should be stopped
		//	(it is becoming too fine or no pattern found so far?!)
		bool			shouldSearchMode(int old_n_candidates = -1) const;

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Current working mode
		Mode 			m_mode;

		// Algorithm for clustering sub-windows
		MeanShiftSelector	m_clusterAlgo;

		// Profile buffers (detection flag + score)
		unsigned char*		m_profileFlags;
		double*			m_profileScores;
	};
}

#endif
