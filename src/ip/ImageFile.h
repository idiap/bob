#ifndef IMAGE_FILE_INC
#define IMAGE_FILE_INC

#include "File.h"

namespace Torch {

	class Image;

	/** This class is designed to handle an image file on the disk.

		@author Sebastien Marcel (marcel@idiap.ch)
		@version 2.0
		\Date
		@since 1.0
	*/
	class ImageFile : public File
	{
	public:
		// Constructor
		ImageFile();

		// Destructor
		virtual ~ImageFile();

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

	protected:

		///////////////////////////////////////////////////////////////////

		// Fills a pixmap from an image object
		static  void		fillPixmap(	unsigned char* pixmap, int n_planes_pixmap,
							const Image& image);

		// Fills an image object from a pixmap (the pixmap is RGB, the image may be gray)
		static void		fillImage(	const unsigned char* pixmap, int n_planes_pixmap,
							Image& image);

		///////////////////////////////////////////////////////////////////
		// Attributes

		//
	};

}

#endif
