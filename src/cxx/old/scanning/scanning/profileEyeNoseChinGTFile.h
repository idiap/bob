/**
 * @file cxx/old/scanning/scanning/profileEyeNoseChinGTFile.h
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
#ifndef _TORCH5SPRO_PROFILE_ENC_GT_FILE_H_
#define _TORCH5SPRO_PROFILE_ENC_GT_FILE_H_

#include "scanning/GTFile.h"

namespace Torch
{
	/**
		Labels: leye_center, nose_tip, chin

	*/
	class profileEyeNoseChinGTFile : public GTFile
	{
	public:
		// Constructor
		profileEyeNoseChinGTFile();

		// Destructor
		virtual ~profileEyeNoseChinGTFile();

		// Load points from some file
		virtual bool		load(File* file);

		// Get the name of the GT file format
		virtual const char*	getName();
	};

}

#endif
