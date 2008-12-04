#ifndef _TORCHVISION_SCANNING_SCANNER_H_
#define _TORCHVISION_SCANNING_SCANNER_H_

#include "Object.h"			// <Scanner> is a <Torch::Object>
#include "Pattern.h"			// detected patterns
#include "vision.h"			// <sRect2D> definition

namespace Torch
{
	class Image;
	class Explorer;
	class Selector;

   	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanner
	//	- scan an image for rectangular patterns in 4D
	// 		(position + scale + model confidence) scanning space!
	//	- uses an Explorer object to investigate the 4D pattern space
	//	- uses a Selector object to select the best patterns from the candidates
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class Scanner : public Torch::Object
	{
	public:

		// Constructor
		Scanner(Explorer* explorer = 0, Selector* selector = 0);

		// Destructor
		virtual ~Scanner();

		/////////////////////////////////////////////////////////////////
		// Modify the ROIs

		bool				addROI(const sRect2D& roi);
		bool				addROI(int x, int y, int w, int h);
		bool				deleteROI(int index);
		void				deleteAllROIs();

		/////////////////////////////////////////////////////////////////

		// Change the explorer
		bool				setExplorer(Explorer* explorer);

		// Change the selector
		bool				setSelector(Selector* selector);

		/////////////////////////////////////////////////////////////////

		// Initialize the scanning (check parameters/objects, initialize explorer)
		bool				init(const Image& image);

		// Process some image to scan for patterns
		bool	 			process(const Image& image);

		/////////////////////////////////////////////////////////////////
		// Access functions

		int				getNoROIs() const { return m_n_rois; }
		const sRect2D&			getROI(int index) const;
		int				getNoScannedSWs() const { return m_stat_scanned; }
		int				getNoPrunnedSWs() const { return m_stat_prunned; }
		int				getNoAcceptedSWs() const { return m_stat_accepted; }
		const PatternList&		getPatterns() const;

	protected:

		/////////////////////////////////////////////////////////////////

		// Check if the scanning parameters are set OK
		virtual bool			checkParameters(const Torch::Image& image) const;

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Object for exploring the 4D scanning space (location + scale + model confidence)
		Explorer*			m_explorer;

		// Object for selecting the (true) patterns from the candidates (pattern space)
		Selector*			m_selector;

		// Regions of interest (ROIs)
		sRect2D*			m_rois;
		int				m_n_rois;

		// Statistical information
		int				m_stat_scanned;	// Number of scanned sub-windows by the pattern model
		int				m_stat_prunned;	// Number of prunned sub-windows
		int				m_stat_accepted;// Number of sub-windows with patterns
			// Total number of investigated sub-windows = m_stat_scanned + m_stat_prunned
			// m_stat_scanned >= m_stat_accepted

	};
}

#endif
