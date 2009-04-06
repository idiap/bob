#ifndef _TORCHVISION_SCANNING_MEAN_SHIFT_SELECTOR_H_
#define _TORCHVISION_SCANNING_MEAN_SHIFT_SELECTOR_H_

#include "Selector.h"		// <MeanShiftSelector> is a <Selector>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::MeanShiftSelector
	//	- merges and selects the final pattern sub-windows using the Mean Shift
	//		clustering algorithm
	//
        // TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class MeanShiftSelector : public Selector
	{
	public:

		// Constructor
		MeanShiftSelector();

		// Destructor
		virtual ~MeanShiftSelector();

		// Delete all stored patterns
		virtual void		clear();

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool		process(const PatternList& candidates);

	private:

		/////////////////////////////////////////////////////////////////

		// Get the closest points to the given one
		// Returns the number of found points
		int			getClosest(const Pattern& pattern);
		int			getClosest(double x, double y, double w, double h);

		static const int 	MaxNoClosestPoints = 1024;

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Buffer to copy the indexes of the closest points
		int 			m_iclosest[MaxNoClosestPoints];
		double			m_iDistClosest[MaxNoClosestPoints];

		// Bandwidths computed for each candidates SW
		bool			m_bandwidthsComputed;
		double*			m_bandwidths;
		double*			m_inv_bandwidths;

		const PatternList*	m_candidates;
	};
}

#endif
