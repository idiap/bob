/**
 * @file cxx/old/scanning/scanning/PyramidExplorer.h
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
#ifndef _TORCHVISION_SCANNING_PYRAMID_EXPLORER_H_
#define _TORCHVISION_SCANNING_PYRAMID_EXPLORER_H_

#include "scanning/Explorer.h"		// <PyramidExplorer> is an <Explorer>

namespace Torch
{
        class ipScaleYX;
        class Tensor;
        class Image;

	/////////////////////////////////////////////////////////////////////////
	// Torch::PyramidExplorerData:
	//	- implementation of <ExplorerData>, just modifying the way
	//		candidate patterns are stored
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	struct PyramidExplorerData : public ExplorerData
	{
		// Constructor
		PyramidExplorerData(ipSWEvaluator* swEvaluator = 0);

		// Destructor
		virtual ~PyramidExplorerData();

		// Store some pattern - just copy it!
		virtual void		storePattern(	int sw_x, int sw_y, int sw_w, int sw_h,
							double confidence);

		// Set the current scanning scale
		void			setScale(const sSize& scale);

		///////////////////////////////////////////////////////////////
		// Attributes

		// Current scaled image size
		sSize			m_scale;
		float			m_inv_scale;
	};

	/////////////////////////////////////////////////////////////////////////
	// Torch::PyramidExplorer
	//	- builds a pyramid of image at different scales, while keeping the model
	//		at a fixed size
	//
	//      - PARAMETERS (name, type, default value, description):
	//		"savePyramidsToJpg"	bool    false   "save the scaled images to JPEG"
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class PyramidExplorer : public Explorer
	{
	public:

		// Constructor
		PyramidExplorer();

		// Destructor
		virtual ~PyramidExplorer();

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

		const Image*		m_image;
	};
}

#endif
