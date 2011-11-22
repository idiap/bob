/**
 * @file cxx/old/scanning/scanning/OverlapSelector.h
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _TORCHVISION_SCANNING_OVERLAP_SELECTOR_H_
#define _TORCHVISION_SCANNING_OVERLAP_SELECTOR_H_

#include "scanning/Selector.h"		// <OverlapSelector> is a <Selector>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::OverlapSelector
	//	- merges and selects the final pattern sub-windows using the overlapping information
	//		~ clustering using the degree of overlapping between candidate sub-windows
	//
        //      - PARAMETERS (name, type, default value, description):
        //		"minSurfOverlap"	int	60	"minimum surface overlapping for merging"
        //		"iterative"		bool	false	"merge the sub-windows in one pass or iteratively"
        //		"onlySurfOverlaps"	bool	false	"keep only sub-windows that overlap with at least another one"
        //		"onlyMaxSurf"		bool	false	"keep only the merged sub-window with the maximum surface"
        //		"onlyMaxConf"		bool	false	"keep only the merged sub-window with the maximum confidence"
        //
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class OverlapSelector : public Selector
	{
	public:

		// Constructor
		OverlapSelector(PatternMerger* merger = 0);

		// Destructor
		virtual ~OverlapSelector();

		// Delete all stored patterns
		virtual void		clear();

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool		process(const PatternList& candidates);

		// Change the merging strategy
		bool			setMerger(PatternMerger* merger);

	private:

		// Resize the buffers to accomodate new candidates
		void			resize(int new_size);

		// Initialize buffers from some pattern list
		void			init(const PatternList& patterns, int minSurfOverlap, bool ignoreInclusion);

		// Deallocate buffers
		void			deallocate();

		// Print the overlapping table
		void			printOverlapTable() const;

		// Overlapping information for a pattern
		struct IndexSurf
		{
			int		m_index;	// Index of the sub-window to which it overlaps
			int		m_surf;		// Overlapping area in percentage
		};
		struct OverlapInfo
		{
			// Constructor
			OverlapInfo(int surf_threshold = 60)
				: 	m_n_overlaps(0),
					m_surf_threshold(surf_threshold)
			{
			}

			// Reset the overlapping information
			void		reset(int surf_threshold)
			{
				m_n_overlaps = 0;
				m_surf_threshold = surf_threshold;
			}

			// Add a new pattern, if it overlaps enough
			void		add(const Pattern& source, const Pattern& test_patt, int index_patt,
						bool ignoreInclusion);

			// Returns the index where to insert a new pattern given its surface overlapping
			// (as to have them ordered descending over the ovelapping surface)
			int		find(int surf, int index_start, int index_stop) const;

			/////////////////////////////////////////////////////////////////

			// The maximum number of overalapping sub-windows to keep for each candidate
			static const int MaxNoOverlaps = 32;

			// Attributes
			IndexSurf	m_overlaps[MaxNoOverlaps];	//
			int		m_n_overlaps;			// The actual number of overlaps used
			int		m_surf_threshold;		//
		};

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Buffers: overlapping table, list of patterns to sort ...
		OverlapInfo*		m_overlappingTable;
		Pattern*		m_patternBuffer;
		unsigned char*		m_patternTags;	// keep track of the merged patterns!
		int			m_n_patterns;
		int			m_n_allocated_patterns;

		// Merging strategy for the overlapping patterns
		PatternMerger*		m_merger;
	};
}

#endif
