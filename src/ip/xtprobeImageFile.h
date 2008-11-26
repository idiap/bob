#ifndef XT_PROBE_IMAGE_FILE_INC
#define XT_PROBE_IMAGE_FILE_INC

#include "ImageFile.h"

namespace Torch {

	/** This class is designed to load an image from the disk using
		the extension of filename to probe the format.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \Date
	    @since 1.0
	*/
	class xtprobeImageFile : public ImageFile
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
		virtual bool 		writePixmap(const Image& pixmap);
		//@}

		/// loads some file and check for its image format - overriden
		virtual bool		open(const char* file_name, const char* open_flags);

	private:

		/////////////////////////////////////////////////////////////////////////////
		// Attributes

		ImageFile*		m_loader;	// The detected image loaded
	};
}

#endif
