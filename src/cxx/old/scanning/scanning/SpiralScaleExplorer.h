/**
 * @file cxx/old/scanning/scanning/SpiralScaleExplorer.h
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
#ifndef _TORCHVISION_SCANNING_SPIRAL_SCALE_EXPLORER_H_
#define _TORCHVISION_SCANNING_SPIRAL_SCALE_EXPLORER_H_

#include "scanning/ScaleExplorer.h"	// <SpiralScaleExplorer> is a <ScaleExplorer>

namespace Torch
{
   	/////////////////////////////////////////////////////////////////////////
	// Torch::SpiralScaleExplorer
	//	- investigates all possible sub-windows at a specified scale
	//		in spiral order from the center of the image,
	//		given some variance on the position (dx and dy parameters)
	//
	//	- PARAMETERS (name, type, default value, description):
	//		"dx"	float	0.1	"OX variation of the pattern width"
	//		"dy"	float	0.1	"OY variation of the pattern height"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class SpiralScaleExplorer : public ScaleExplorer
	{
	public:

		// Constructor
		SpiralScaleExplorer();

		// Destructor
		virtual ~SpiralScaleExplorer();

		// Process the scale, searching for patterns at different sub-windows
		virtual bool		process(ExplorerData& explorerData,
						bool stopAtFirstDetecton);

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
