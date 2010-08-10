#ifndef GIF_IMAGE_FILE_INC
#define GIF_IMAGE_FILE_INC

#include "ip/ImageFile.h"

namespace Torch {

	#define MAXCOLORMAPSIZE         256

	struct gifscreen
	{
		unsigned int    Width;
		unsigned int    Height;
		unsigned char   ColorMap[3][MAXCOLORMAPSIZE];
		unsigned int    BitPixel;
		unsigned int    ColorResolution;
		unsigned int    Background;
		unsigned int    AspectRatio;
	};

	struct gif89
	{
		int transparent;
		int delayTime;
		int inputFlag;
		int disposal;
	};

	/** This class is designed to handle a GIF image file on the disk

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class gifImageFile : public ImageFile
	{
	public:
		// Constructor
		gifImageFile();

		// Destructor
		virtual ~gifImageFile();

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

		/////////////////////////////////////////////////////////////////
		// Functions for decoding GIF format

		int ReadColorMap(int number, unsigned char buffer[3][MAXCOLORMAPSIZE]);
		int DoExtension(int label);
		int GetDataBlock(unsigned char* buf);
		int GetCode(int code_size, int flag);
		int LWZReadByte(int flag, int input_code_size);

		bool read_rgbimage_from_gif(unsigned char* pixmap,
				int width, int height, unsigned char cmap[3][MAXCOLORMAPSIZE], int interlace);

		/////////////////////////////////////////////////////////////////
		// Attributes

		// GIF specific structures
		gifscreen  		GifScreen;
		gif89     		Gif89;
		int 			imageNumber;
		int 			ZeroDataBlock;
	};

}

#endif
