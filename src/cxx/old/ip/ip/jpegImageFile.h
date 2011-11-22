/**
 * @file cxx/old/ip/ip/jpegImageFile.h
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
#ifndef JPEG_IMAGE_FILE_INC
#define JPEG_IMAGE_FILE_INC

#include "ip/ImageFile.h"

#ifdef HAVE_JPEG
extern "C"
{
	#include <jpeglib.h>
	#include <setjmp.h>
}
#endif

namespace Torch {

	/** This class is designed to handle a JPEG image on the disk

	    JPEG source code can be found \URL[here]{http://www.ijg.org/files/jpegsrc.v6b.tar.gz}.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class jpegImageFile : public ImageFile
	{
	public:
		// Constructor
		jpegImageFile();

		// Destructor
		virtual ~jpegImageFile();

	protected:

		/**@name read/write image header and image pixmap */
		//@{
		/** read the image header from the file.
		*/
		virtual bool 		readHeader(Image& image);

		/// read the image pixmap from the file
		virtual bool 		readPixmap(Image& image);

		/** write the image header to a file.
		*/
		virtual bool 		writeHeader(const Image& image);

		/// write the #pixmap_# into the file
		virtual bool 		writePixmap(const Image& image);
		//@}

	private:

		///////////////////////////////////////////////////////////////////////
		// Attributes

		//
	};
}

#endif
