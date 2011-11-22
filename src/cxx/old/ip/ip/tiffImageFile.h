/**
 * @file cxx/old/ip/ip/tiffImageFile.h
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
#ifndef TIFF_IMAGE_FILE_INC
#define TIFF_IMAGE_FILE_INC

#include "ip/ImageFile.h"

namespace Torch {

#ifdef HAVE_TIFF
#include <tiffio.h>
#endif

	/** This class is designed to handle a TIFF image on the disk

	    TIFF source code can be found \URL[here]{http://www.libtiff.org/}.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class tiffImageFile : public ImageFile
	{
	public:
		// Constructor
		tiffImageFile();

		// Destructor
		virtual ~tiffImageFile();

		// Save an image - overriden
		virtual bool		save(const Image& image, const char* filename);

		// Load an image - overriden
		virtual bool		load(Image& image, const char* filename);

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

	#ifdef HAVE_TIFF
		///////////////////////////////////////////////////////////////////////////////
		// Specific TIFF decoding functions

		void 			unpack_tiff_raster(uint32* raster, unsigned char* pixmap);

		///////////////////////////////////////////////////////////////////////////////
		// Attribues

		TIFF *	tif;
		short bpp;
		int scanline_size;
	#endif
	};
}

#endif
