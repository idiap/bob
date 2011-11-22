/**
 * @file cxx/old/scanning/scanning/bbx2eye19x19deye10GTFile.h
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
#ifndef _TORCH5SPRO_BBX2EYE19X19DEYE10_GT_FILE_H_
#define _TORCH5SPRO_BBX2EYE19X19DEYE10_GT_FILE_H_

#include "scanning/GTFile.h"
#include "scanning/Pattern.h"

namespace Torch
{
	/**
		Labels: leye_center, reye_center

	*/
	class bbx2eye19x19deye10_GTFile : public GTFile
	{
	public:
		// Constructor
		bbx2eye19x19deye10_GTFile();

		// Destructor
		virtual ~bbx2eye19x19deye10_GTFile();

		// Load points from some file
		virtual bool		load(File* file);

    /**
     * Load points already in memory
     * @param p The pattern containing the bounding box you want to use
     *
     * @return Always true...
     */
    virtual bool    load(const Torch::Pattern& p);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
