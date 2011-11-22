/**
 * @file cxx/old/ip/ip/xtprobeImageFile.h
 * @date Sat Apr 30 18:41:25 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#ifndef XT_PROBE_IMAGE_FILE_INC
#define XT_PROBE_IMAGE_FILE_INC

#include "ip/ImageFile.h"

namespace Torch {

	class ImageFile;

	/** This class is designed to load an image from the disk using
		the extension of filename to probe the format.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class xtprobeImageFile
	{
	public:
		/**@name constructor and destructor */
		//@{
		/** Makes a ImageFile.
		*/
		xtprobeImageFile();

		/// Destructor
		virtual ~xtprobeImageFile();
		//@}

		// Save an image
		bool			save(const Image& image, const char* filename);

		// Load an image
		bool			load(Image& image, const char* filename);

	private:

		/////////////////////////////////////////////////////////////////////////////
		// Attributes

		ImageFile*		m_loader;	// The detected image loaded
	};
}

#endif
