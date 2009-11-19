#ifndef _TORCH5SPRO_IMAGE_FILE_H_
#define _TORCH5SPRO_IMAGE_FILE_H_

#include "File.h"

namespace Torch
{
	class Image;

	/** This class is designed to handle an image file on the disk.

		@author Sebastien Marcel (marcel@idiap.ch)
		@version 2.0
		\Date
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
