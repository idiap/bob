#include "ip/gifImageFile.h"
#include "ip/Image.h"

namespace Torch {

#define CM_RED                  0
#define CM_GREEN                1
#define CM_BLUE                 2

#define MAX_LWZ_BITS            12

#define INTERLACE               0x40
#define LOCALCOLORMAP           0x80

#define BitSet(byte, bit)       (((byte) & (bit)) == (bit))

#define LM_to_uint(a,b)         (((b)<<8)|(a))

//////////////////////////////////////////////////////////////////////////////////////
// Constructor

gifImageFile::gifImageFile()
{
	Gif89.transparent = -1;
        Gif89.delayTime = -1;
        Gif89.inputFlag = -1;
        Gif89.disposal = 0;
	ZeroDataBlock = false;

	imageNumber = 1;
}

//////////////////////////////////////////////////////////////////////////////////////
// Destructor

gifImageFile::~gifImageFile()
{
}

//////////////////////////////////////////////////////////////////////////////////////
// Read the image header

bool gifImageFile::readHeader(Image& image)
{
   	unsigned char	buf[16];
	char		version[4];

	if(!m_file.read(buf, 6, 1))
	{
		print("gifImageFile::readHeader - reading magic number\n");
		return false;
	}

	if(strncmp((const char *) buf, "GIF",3) != 0)
	{
		print("gifImageFile::readHeader - not a GIF file\n");
		return false;
	}

	strncpy(version, (const char *) (buf + 3), 3);
	version[3] = '\0';

	if((strcmp(version, "87a") != 0) && (strcmp(version, "89a") != 0))
	{
		print("gifImageFile::readHeader - bad version number, not '87a' or '89a'\n");
		return false;
	}

	if(!m_file.read(buf, 7, 1))
	{
		print("gifImageFile::readHeader - failed to read screen descriptor\n");
		return false;
	}

	GifScreen.Width           = LM_to_uint(buf[0],buf[1]);
	GifScreen.Height          = LM_to_uint(buf[2],buf[3]);
	GifScreen.BitPixel        = 2<<(buf[4]&0x07);
	GifScreen.ColorResolution = (((buf[4]&0x70)>>3)+1);
	GifScreen.Background      = buf[5];
	GifScreen.AspectRatio     = buf[6];

	const int width = GifScreen.Width;
	const int height = GifScreen.Height;

	if(BitSet(buf[4], LOCALCOLORMAP))
	{	/* Global Colormap */
		if (ReadColorMap(GifScreen.BitPixel, GifScreen.ColorMap))
		{
			print("gifImageFile::readHeader - error reading global colormap\n");
			return false;
		}
	}

	if(GifScreen.AspectRatio != 0 && GifScreen.AspectRatio != 49)
	{
		float	r;
		r = ( (float) GifScreen.AspectRatio + 15.0 ) / 64.0;

		warning("gifImageFile::readHeader - non-square pixels; to fix do a 'pnmscale -%cscale %g'\n",
		    r < 1.0 ? 'x' : 'y',
		    r < 1.0 ? 1.0 / r : r );
	}

	// OK, resize the image to the new dimensions
	return image.resize(width, height, image.getNPlanes());// Keep the number of color channels
}

//////////////////////////////////////////////////////////////////////////////////////
// Read the image pixmap

bool gifImageFile::readPixmap(Image& image)
{
	const int width = image.getWidth();
	const int height = image.getHeight();

	unsigned char* pixmap = new unsigned char[3 * width * height];

	const bool verbose = false;

	unsigned char	buf[16];
	unsigned char	c;
	unsigned char	localColorMap[3][MAXCOLORMAPSIZE];
	int		useGlobalColormap;
	int		bitPixel;
	int		imageCount = 0;

	for(;;)
	{
		if(!m_file.read(&c, 1, 1))
		{
			print("gifImageFile::readPixmap - EOF / read error on image data\n");
			delete[] pixmap;
			return false;
		}

		if(c == ';')
		{	/* GIF terminator */
			if (imageCount < imageNumber)
			{
				print("gifImageFile::readPixmap - only %d image%s found in file\n",
					imageCount, imageCount>1?"s":"" );
				delete[] pixmap;
				return false;
			}
			return true;
		}

		if(c == '!')
		{ 	/* Extension */
			if(!m_file.read(&c, 1, 1))
			{
				print("gifImageFile::readPixmap - OF / read error on extention function code\n");
				delete[] pixmap;
				return false;
			}

			DoExtension(c);
			continue;
		}

		if(c != ',')
		{	/* Not a valid start character */
			message("gifImageFile::readPixmap - bogus character 0x%02x, ignoring\n", (int) c);
			continue;
		}

		++imageCount;

		if(!m_file.read(buf, 9, 1))
		{
			print("gifImageFile::readPixmap - couldn't read left/top/width/height\n");
			delete[] pixmap;
			return false;
		}

		useGlobalColormap = ! BitSet(buf[8], LOCALCOLORMAP);

		bitPixel = 1<<((buf[8]&0x07)+1);

		int w = LM_to_uint(buf[4],buf[5]);
		int h = LM_to_uint(buf[6],buf[7]);

		if(verbose)
			print("gifImageFile::readPixmap - GIF(%d): %d x %d\n", imageCount, w, h);

		if(!useGlobalColormap)
		{
			if(ReadColorMap(bitPixel, localColorMap))
			{
				print("gifImageFile::readPixmap - error reading local colormap\n");
				delete[] pixmap;
				return false;
			}

			if(read_rgbimage_from_gif(pixmap, width, height,
						localColorMap, BitSet(buf[8], INTERLACE)) == false)
			{
				delete[] pixmap;
				return false;
			}
		}
		else
		{
			if(read_rgbimage_from_gif(pixmap, width, height,
					GifScreen.ColorMap, BitSet(buf[8], INTERLACE)) == false)
			{
				delete[] pixmap;
				return false;
			}
		}

		// Force to finish to read only 1 picture from gif file
		break;
	}

	// OK, update the image pixels to the pixmap
	Image::fillImage(pixmap, 3, image);
	delete[] pixmap;
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////
// Write the image header

bool gifImageFile::writeHeader(const Image& image)
{
	warning("gifImageFile::writeHeader - this feature is not implemented.\n");
	return false;
}

//////////////////////////////////////////////////////////////////////////////////////
// Write the image pixmap

bool gifImageFile::writePixmap(const Image& image)
{
	warning("gifImageFile::writePixmap - this feature is not implemented.\n");
	return false;
}

//////////////////////////////////////////////////////////////////////////////////////
// Functions for decoding GIF format

int gifImageFile::ReadColorMap(int number, unsigned char buffer[3][MAXCOLORMAPSIZE])
{
	unsigned char rgb[3];

	for (int i = 0; i < number; ++i)
	{
		if(!m_file.read(rgb, sizeof(rgb), 1))
			print("Error: bad colormap\n");

		buffer[CM_RED][i] = rgb[0];
		buffer[CM_GREEN][i] = rgb[1];
		buffer[CM_BLUE][i] = rgb[2];
	}

	return 0;
}

int gifImageFile::DoExtension(int label)
{
	static char buf[256];
	const char *str;

	const bool verbose = false;

	switch (label)
	{
	case 0x01:
	   	/* Plain Text Extension */
		str = "Plain Text Extension";
		break;
	case 0xff:
		/* Application Extension */
		str = "Application Extension";
		break;
	case 0xfe:
		/* Comment Extension */
		str = "Comment Extension";
		while(GetDataBlock((unsigned char*) buf) != 0)
			if(verbose) message("gif comment: %s", buf);
		return 0;
	case 0xf9:
		/* Graphic Control Extension */
		str = "Graphic Control Extension";
		(void) GetDataBlock((unsigned char*) buf);
		Gif89.disposal    = (buf[0] >> 2) & 0x7;
		Gif89.inputFlag   = (buf[0] >> 1) & 0x1;
		Gif89.delayTime   = LM_to_uint(buf[1],buf[2]);
		if ((buf[0] & 0x1) != 0)
			Gif89.transparent = buf[3];
		while (GetDataBlock((unsigned char*) buf) != 0);

		return 0;
	default:
		str = buf;
		sprintf(buf, "UNKNOWN (0x%02x)", label);
		break;
	}

	if(verbose)
		message("got a '%s' extension", str);

	while (GetDataBlock((unsigned char*) buf) != 0);

	return 0;
}

int gifImageFile::GetDataBlock(unsigned char *buf)
{
	unsigned char count;

	if(!m_file.read(&count, 1, 1))
	{
		print("Error: getting DataBlock size\n");

		return -1;
	}

	ZeroDataBlock = count == 0;

	if ((count != 0) && (!m_file.read(buf, count, 1)))
	{
		print("Error: reading DataBlock\n");
		return -1;
	}

	return count;
}

int gifImageFile::GetCode(int code_size, int flag)
{
	static unsigned char buf[280];
	static int curbit, lastbit, done, last_byte;
	int i, j, ret;
	unsigned char count;

	if(flag)
	{
		curbit = 0;
		lastbit = 0;
		done = false;
		return 0;
	}

	if((curbit+code_size) >= lastbit)
	{
		if(done)
		{
			if (curbit >= lastbit)
				print("Error: ran off the end of my bits\n");
			return -1;
		}
		buf[0] = buf[last_byte-2];
		buf[1] = buf[last_byte-1];

		if ((count = GetDataBlock(&buf[2])) == 0)
			done = true;

		last_byte = 2 + count;
		curbit = (curbit - lastbit) + 16;
		lastbit = (2+count)*8 ;
	}

	ret = 0;
	for (i = curbit, j = 0; j < code_size; ++i, ++j)
		ret |= ((buf[ i / 8 ] & (1 << (i % 8))) != 0) << j;

	curbit += code_size;

	return ret;
}

int gifImageFile::LWZReadByte(int flag, int input_code_size)
{
	static int	fresh = false;
	int		code, incode;
	static int	code_size, set_code_size;
	static int	max_code, max_code_size;
	static int	firstcode, oldcode;
	static int	clear_code, end_code;
	static int	table[2][(1<< MAX_LWZ_BITS)];
	static int	stack[(1<<(MAX_LWZ_BITS))*2], *sp;
	register int	i;

	if (flag)
	{
		set_code_size = input_code_size;
		code_size = set_code_size+1;
		clear_code = 1 << set_code_size ;
		end_code = clear_code + 1;
		max_code_size = 2*clear_code;
		max_code = clear_code+2;

		GetCode(0, true);

		fresh = true;

		for (i = 0; i < clear_code; ++i)
		{
			table[0][i] = 0;
			table[1][i] = i;
		}
		for (; i < (1<<MAX_LWZ_BITS); ++i)
			table[0][i] = table[1][0] = 0;

		sp = stack;

		return 0;
	}
	else if (fresh)
	{
		fresh = false;
		do {
			firstcode = oldcode =
				GetCode(code_size, false);
		} while (firstcode == clear_code);
		return firstcode;
	}

	if (sp > stack)
		return *--sp;

	while ((code = GetCode(code_size, false)) >= 0)
	{
		if (code == clear_code)
		{
			for (i = 0; i < clear_code; ++i)
			{
				table[0][i] = 0;
				table[1][i] = i;
			}
			for (; i < (1<<MAX_LWZ_BITS); ++i)
				table[0][i] = table[1][i] = 0;
			code_size = set_code_size+1;
			max_code_size = 2*clear_code;
			max_code = clear_code+2;
			sp = stack;
			firstcode = oldcode =
					GetCode(code_size, false);
			return firstcode;
		}
		else if (code == end_code)
		{
			int		count;
			unsigned char	buf[260];

			if (ZeroDataBlock)
				return -2;

			while ((count = GetDataBlock(buf)) > 0)
				;

			if (count != 0)
				message("missing EOD in data stream (common occurence)");
			return -2;
		}

		incode = code;

		if (code >= max_code)
		{
			*sp++ = firstcode;
			code = oldcode;
		}

		while (code >= clear_code)
		{
			*sp++ = table[1][code];
			if (code == table[0][code])
				print("Error: circular table entry BIG ERROR\n");
			code = table[0][code];
		}

		*sp++ = firstcode = table[1][code];

		if ((code = max_code) <(1<<MAX_LWZ_BITS))
		{
			table[0][code] = oldcode;
			table[1][code] = firstcode;
			++max_code;
			if ((max_code >= max_code_size) &&
				(max_code_size < (1<<MAX_LWZ_BITS)))
			{
				max_code_size *= 2;
				++code_size;
			}
		}

		oldcode = incode;

		if (sp > stack)
			return *--sp;
	}
	return code;
}

bool gifImageFile::read_rgbimage_from_gif(unsigned char* pixmap,
		int width, int height, unsigned char cmap[3][MAXCOLORMAPSIZE], int interlace)
{
	unsigned char	c;
	int		v;
	int		xpos = 0, ypos = 0, pass = 0;

	/*
	**  Initialize the Compression routines
	*/
	if(!m_file.read(&c, 1, 1))
	{
		print("Error: EOF / read error on image data\n");

		return false;
	}

	if(LWZReadByte(true, c) < 0)
	{
		print("Error: reading image\n");

		return false;
	}

	const bool verbose = false;
	if (verbose)
		message("reading %d by %d%s GIF image",
			width, height, interlace ? " interlaced" : "" );

	while ((v = LWZReadByte(false,c)) >= 0 )
	{
		int offset = xpos + ypos * width;

		pixmap[3*offset] = cmap[CM_RED][v];
		pixmap[3*offset+1] = cmap[CM_GREEN][v];
		pixmap[3*offset+2] = cmap[CM_BLUE][v];

		++xpos;
		if (xpos == width)
		{
			xpos = 0;
			if (interlace)
			{
				switch (pass)
				{
				case 0:
				case 1:
					ypos += 8; break;
				case 2:
					ypos += 4; break;
				case 3:
					ypos += 2; break;
				}

				if (ypos >= height)
				{
					++pass;
					switch (pass)
					{
					case 1:
						ypos = 4; break;
					case 2:
						ypos = 2; break;
					case 3:
						ypos = 1; break;
					default:
						goto end_of_read_rgbimage_from_gif;
					}
				}
			}
			else
			{
				++ypos;
			}
		}
		if (ypos >= height)
			break;
	}

end_of_read_rgbimage_from_gif:

	if (LWZReadByte(false,c)>=0)
		print("too much input data, ignoring extra...\n");

	return true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}
