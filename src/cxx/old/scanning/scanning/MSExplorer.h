/**
 * @file cxx/old/scanning/scanning/MSExplorer.h
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
#ifndef _TORCHVISION_SCANNING_MS_EXPLORER_H_
#define _TORCHVISION_SCANNING_MS_EXPLORER_H_

#include "scanning/Explorer.h"		// <MSExplorer> is an <Explorer>

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::MSExplorerData:
	//	- implementation of <ExplorerData>, just modifying the way
	//		candidate patterns are stored
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct MSExplorerData : public ExplorerData
	{
		// Constructor
		MSExplorerData(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~MSExplorerData();

		// Store some pattern - just copy it!
		virtual void		storePattern(	int sw_x, int sw_y, int sw_w, int sw_h,
							double confidence);
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::MSExplorer
	//	- MultiScale explorer
	//	- keep the image size and vary the size of model/template (window scan)
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class MSExplorer : public Explorer
	{
	public:
		// Constructor
		MSExplorer();

		// Destructor
		virtual ~MSExplorer();

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

		// Initialize the scanning process with the given image size
		virtual bool		init(int image_w, int image_h);

		// Initialize the scanning process for a specific ROI
		virtual bool		init(const sRect2D& roi);

		// Preprocess the image (extract features ...) => store data in <prune_ips> and <evaluation_ips>
		virtual bool		preprocess(const Image& image);

		// Process the image (check for pattern's sub-windows)
		virtual bool		process();

		/////////////////////////////////////////////////////////////////

	protected:

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Tensors (features) for the pruning and evaluation
		const Tensor*           m_prune_tensor;
		const Tensor*           m_evaluation_tensor;
	};
}

#endif
