#ifndef _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_
#define _TORCHVISION_SCANNING_GREEDY_EXPLORER_H_

#include "MSExplorer.h"		// <GreedyExplorer> is a <MSExplorer>
#include "ProfileMachine.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::GreedyExplorer
	//	- searches the 4D scanning space using a greedy method based on
	//		a context-based (profile) model to remove false alarms
	//		and drive iteratively the detections to better locations
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"ctx_type"	int	1	"0 - full context, 1 - axis context"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class GreedyExplorer : public MSExplorer
	{
	public:

		// Running mode
		enum Mode
		{
			Scanning,	// Greedy scanning (using the context-based model)
			Profiling	// MS scanning (for generating profiles)
		};

		// Context type
		enum ContextType
		{
			Full,
			Axis
		};

		// Constructor
		GreedyExplorer(Mode = Scanning);

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

		// Set the context classifier
		bool			setContextModel(const char* filename);

		// Access functions
		void			setMode(Mode mode) { m_mode = mode; }
		Mode 			getMode() const { return m_mode; }
		const unsigned char*	getContextFlags() const { return m_ctx_flags; }
		const double*		getContextScores() const { return m_ctx_scores; }
		int			getContextOx(int index) const { return m_ctx_ox[index]; }
		int			getContextOy(int index) const { return m_ctx_oy[index]; }
		int			getContextOs(int index) const { return m_ctx_os[index]; }
		int			getContextSize() const { return m_ctx_size; }

		/////////////////////////////////////////////////////////////////

	protected:

		// Add a subwindow to the profile
		void			addSWToProfile(	int sw_x, int sw_y, int sw_w, int sw_h,
							unsigned char& flag, double& score);

		/// called when some option was changed
		virtual void		optionChanged(const char* name);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Current working mode
		Mode 			m_mode;		// Scanning (MS + context model), Profiling (MS)

		// Context model, type, buffers
		ProfileMachine		m_ctx_model;
		ContextType		m_ctx_type;
		int			m_ctx_size;
		unsigned char*		m_ctx_flags;
		double*			m_ctx_scores;
		int*			m_ctx_ox;
		int*			m_ctx_oy;
		int*			m_ctx_os;
		PatternMerger*		m_ctx_sw_merger;
	};
}

#endif
