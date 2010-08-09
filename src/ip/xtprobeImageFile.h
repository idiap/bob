#ifndef XT_PROBE_IMAGE_FILE_INC
#define XT_PROBE_IMAGE_FILE_INC

#include "ImageFile.h"

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
