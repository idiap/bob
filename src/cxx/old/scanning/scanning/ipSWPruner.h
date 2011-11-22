/**
 * @file cxx/old/scanning/scanning/ipSWPruner.h
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
#ifndef _TORCHVISION_SCANNING_IP_SW_PRUNER_H_
#define _TORCHVISION_SCANNING_IP_SW_PRUNER_H_

#include "ip/ipCore.h"		// <ipSWPruner> is an <ipCore>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::ipSWPruner
	//	- rejects some sub-window (e.g. based on the pixel/edge variance
	//		- too smooth or too noisy)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ipSWPruner : public ipCore
	{
	public:

		// Constructor
		ipSWPruner();

		// Destructor
		virtual ~ipSWPruner();

		// Get the result - the sub-window is rejected?!
		bool			isRejected() const { return m_isRejected; }

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		bool			m_isRejected;
	};
}

#endif
