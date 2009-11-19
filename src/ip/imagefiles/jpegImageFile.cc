#include "jpegImageFile.h"
#include "Image.h"

namespace Torch {

#ifdef HAVE_JPEG

/////////////////////////////////////////////////////////////
// Error handlers for jpeg library

struct my_jpeg_error_mgr : public jpeg_error_mgr
{
	jmp_buf setjmp_buffer;
};

static void my_jpeg_error_exit(j_common_ptr cinfo);
static void my_jpeg_emit_message(j_common_ptr cinfo, int msg_level);
static void my_jpeg_output_message(j_common_ptr cinfo);

static void my_jpeg_error_exit(j_common_ptr cinfo)
{
	my_jpeg_error_mgr* myerr = (my_jpeg_error_mgr*) cinfo->err;

	//char buffer[JMSG_LENGTH_MAX];
	//(*cinfo->err->format_message)(cinfo, buffer);
	//Torch::message(buffer);

	//(*cinfo->err->output_message) (cinfo);

	longjmp(myerr->setjmp_buffer, 1);
}

static void my_jpeg_emit_message(j_common_ptr cinfo, int msg_level)
{
	//char buffer[JMSG_LENGTH_MAX];
	//(*cinfo->err->format_message)(cinfo, buffer);

	//Torch::message(buffer);
}

static void my_jpeg_output_message(j_common_ptr cinfo)
{
	//char buffer[JMSG_LENGTH_MAX];
	//(*cinfo->err->format_message)(cinfo, buffer);

	//Torch::message(buffer);
}

////////////////////////////////////////////////////////////////////////////////////////

#endif

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

jpegImageFile::jpegImageFile()
	:	ImageFile()
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

jpegImageFile::~jpegImageFile()
{
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Read the image header

bool jpegImageFile::readHeader(Image& image)
{
#ifdef HAVE_JPEG

	// The decoding will be done in the <readPixmap>!
	return true;

#else
	warning("jpegDiskXFile::readHeader(). JPEG format not supported.");
	return false;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Read the image pixmap

bool jpegImageFile::readPixmap(Image& image)
{
#ifdef HAVE_JPEG

	jpeg_decompress_struct 	dinfo;

	my_jpeg_error_mgr jerr;
	dinfo.err                 = jpeg_std_error(&jerr);
	dinfo.err->error_exit     = my_jpeg_error_exit;
	dinfo.err->emit_message   = my_jpeg_emit_message;
	dinfo.err->output_message = my_jpeg_output_message;

	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_decompress(&dinfo);
		return false;
	}

	jpeg_create_decompress(&dinfo);

	if (m_file.m_file != NULL)
	{
		jpeg_stdio_src(&dinfo, m_file.m_file);
	}
	else
	{
		Torch::message("jpegImageFile::readHeader - the file is not opened!\n");
		jpeg_destroy_decompress(&dinfo);
		return false;
	}

	jpeg_read_header(&dinfo, true);
        jpeg_start_decompress(&dinfo);

        if (	dinfo.output_components != 3 &&
		dinfo.output_components != 1)
	{
		Torch::message("jpegImageFile::readHeader - this is not a gray level one!\n");
		jpeg_destroy_decompress(&dinfo);
		return false;
	}

	// OK, resize the image to the new dimensions
	const int width = dinfo.output_width;
	const int height = dinfo.output_height;
	if (image.resize(width, height, image.getNPlanes()) == false)// Keep the number of color channels
	{
		jpeg_destroy_decompress(&dinfo);
		return false;
	}

	// Decode the image into a pixmap
	unsigned char* pixmap = new unsigned char[3 * width * height];

	const int row_stride = dinfo.output_width * dinfo.output_components;
	JSAMPARRAY buffer =
		(*dinfo.mem->alloc_sarray) ((j_common_ptr) &dinfo, JPOOL_IMAGE, row_stride, 1);

	int i = 0;
        while(dinfo.output_scanline < dinfo.output_height)
        {
                (void) jpeg_read_scanlines(&dinfo, buffer, 1);

		JSAMPLE* line = buffer[0];

                int j = 0;
                while(j < row_stride)
                {
		   	if(dinfo.output_components == 1)
			{
                        	pixmap[i++] = (unsigned char) line[j];
                        	pixmap[i++] = (unsigned char) line[j];
	                        pixmap[i++] = (unsigned char) line[j++];
			}
			else
			{
                        	pixmap[i++] = (unsigned char) line[j++];
                        	pixmap[i++] = (unsigned char) line[j++];
	                        pixmap[i++] = (unsigned char) line[j++];
			}
                }
        }

	// Finish the decoding
        (void) jpeg_finish_decompress(&dinfo);
        jpeg_destroy_decompress(&dinfo);

        // OK, update the image pixels to the pixmap
	Image::fillImage(pixmap, 3, image);
	delete[] pixmap;
	return true;
#else
	warning("jpegImageFile::readPixmap(). JPEG format not supported.");
	return false;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Write the image header

bool jpegImageFile::writeHeader(const Image& image)
{
#ifdef HAVE_JPEG

	// The encoding will be done in <writePixmap>
	return true;
#else
	warning("jpegImageFile::writeHeader(). JPEG format not supported.");
	return false;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
// Write the image pixmap

bool jpegImageFile::writePixmap(const Image& image)
{
#ifdef HAVE_JPEG

	jpeg_compress_struct cinfo;
	const int width = image.getWidth();
   	const int height = image.getHeight();

   	struct my_jpeg_error_mgr       jerr;
	cinfo.err                 = jpeg_std_error(&jerr);
	cinfo.err->error_exit     = my_jpeg_error_exit;
	cinfo.err->emit_message   = my_jpeg_emit_message;
	cinfo.err->output_message = my_jpeg_output_message;

	if (setjmp(jerr.setjmp_buffer))
	{
		jpeg_destroy_compress(&cinfo);
		return false;
	}

	// Initialize the encoding
	jpeg_create_compress(&cinfo);
	jpeg_stdio_dest(&cinfo, m_file.m_file);

	int quality = 95;
  	jpeg_set_quality(&cinfo, quality, true);

	cinfo.image_width = width; 	/* image width and height, in pixels */
  	cinfo.image_height = height;
  	cinfo.input_components = 3;	/* # of color components per pixel */
  	cinfo.in_color_space = JCS_RGB; /* colorspace of input image */
	cinfo.optimize_coding = 100;
        cinfo.smoothing_factor = 0;

  	jpeg_set_defaults(&cinfo);
  	jpeg_start_compress(&cinfo, true);

	// Fill the pixmap with the image pixels
	unsigned char* pixmap = new unsigned char[3 * width * height];
	Image::fillPixmap(pixmap, 3, image);

	// Encode the pixmap
	const int row_stride = width * 3;
   	JSAMPROW row_pointer[1];

   	while(cinfo.next_scanline < cinfo.image_height)
	{
    		row_pointer[0] = &pixmap[cinfo.next_scanline * row_stride];
    		(void) jpeg_write_scanlines(&cinfo, row_pointer, 1);
  	}

  	jpeg_finish_compress(&cinfo);
  	jpeg_destroy_compress(&cinfo);

	// OK
	delete[] pixmap;
	return true;
#else
	warning("jpegImageFile::writePixmap(). JPEG format not supported.");
	return false;
#endif
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

}
