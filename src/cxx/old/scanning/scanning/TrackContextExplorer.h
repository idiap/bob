/**
 * @file cxx/old/scanning/scanning/TrackContextExplorer.h
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
#ifndef _TORCHVISION_SCANNING_TRACK_CONTEXT_EXPLORER_H_
#define _TORCHVISION_SCANNING_TRACK_CONTEXT_EXPLORER_H_

#include "scanning/ContextExplorer.h"		// <TrackContextExplorer> is a <ContextExplorer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::TrackContextExplorer
	//	- process only a specified target sub-window
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class TrackContextExplorer : public ContextExplorer
	{
	public:

		// Constructor
		TrackContextExplorer(Mode mode = Scanning);

		// Destructor
		virtual ~TrackContextExplorer();
		
		// Change the sub-windows to process
		void setSeedPatterns(const PatternList& patterns);
		
	protected:
	  
		// Initialize the sub-windows to process
		virtual bool		initContext();

		/////////////////////////////////////////////////////////////////
		// Attributes
	
		PatternList		m_seed_patterns;
	};
}

#endif
