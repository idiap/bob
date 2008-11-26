#ifndef JPEG_IMAGE_FILE_INC
#define JPEG_IMAGE_FILE_INC

#include "ImageFile.h"

namespace Torch {

#ifdef HAVE_JPEG
extern "C"
{
	#include <jpeglib.h>
	#include <setjmp.h>
}
#endif

	/** This class is designed to handle a JPEG image on the disk

	    JPEG source code can be found \URL[here]{http://www.ijg.org/files/jpegsrc.v6b.tar.gz}.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \Date
	    @since 1.0
	*/
	class jpegImageFile : public ImageFile
	{
	public:
		// Constructor
		jpegImageFile();

		// Destructor
		virtual ~jpegImageFile();

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
