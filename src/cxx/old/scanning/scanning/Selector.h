/**
 * @file cxx/old/scanning/scanning/Selector.h
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
#ifndef _TORCHVISION_SCANNING_SELECTOR_H_
#define _TORCHVISION_SCANNING_SELECTOR_H_

#include "core/Object.h"		// <Selector> is a <Torch::Object>
#include "scanning/Pattern.h"		// works on <Pattern>s

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::Scanning::Selector
	//	- given a list of candidate sub-windows,
	//		it will select just some of these sub-windows as the best ones!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class Selector : public Torch::Object
	{
	public:

		// Constructor
		Selector();

		// Destructor
		virtual ~Selector();

		// Delete all stored patterns
		virtual void			clear() { m_patterns.clear(); }

		// Process the list of candidate sub-windows and select the best ones
		// (this will accumulate them to the pattern list)
		virtual bool			process(const PatternList& candidates) = 0;

		// Return the result
		const PatternList&		getPatterns() const { return m_patterns; }

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Result: best pattern selected from the candidates (the 4D scanning space)
		PatternList			m_patterns;
	};

	/////////////////////////////////////////////////////////////////////////
}

#endif
