/**
 * @file cxx/old/ip/ip/ImageFile.h
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
#ifndef _TORCH5SPRO_IMAGE_FILE_H_
#define _TORCH5SPRO_IMAGE_FILE_H_

#include "core/File.h"

namespace Torch
{
	class Image;

	/** This class is designed to handle an image file on the disk.

		@author Sebastien Marcel (marcel@idiap.ch)
		@version 2.0
		\date
		@since 1.0
	*/
	class ImageFile
	{
	public:
		// Constructor
		ImageFile();

		// Destructor
		virtual ~ImageFile();

		// Save an image
		virtual bool		save(const Image& image, const char* filename);

		// Load an image
		virtual bool		load(Image& image, const char* filename);

	protected:

		/**@name read/write image header and image pixmap */
		//@{
		/** read the image header from the file.
		*/
		virtual bool 		readHeader(Image& image) = 0;

		/// read the image pixmap from the file
		virtual bool 		readPixmap(Image& image) = 0;

		/** write the image header to a file.
		*/
		virtual bool 		writeHeader(const Image& image) = 0;

		/// write the #pixmap_# into the file
		virtual bool 		writePixmap(const Image& image) = 0;
		//@}

		///////////////////////////////////////////////////////////////////
		// Attributes

		File			m_file;
	};

}

#endif
