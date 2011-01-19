#ifndef COLOR_INC
#define COLOR_INC

#include "ip/Convert.h"

namespace Torch {

	/** This class is designed to handle colors

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Color
	{
	public:
		/**@name fields */
		//@{
		/// the R or Y color component
		unsigned char data0;

		/// the G or U color component
		unsigned char data1;

		/// the B or V color component
		unsigned char data2;

		/// the name of the colorspace ("rgb" or "yuv")
		const char *coding;
		//@}

		//-----

		/**@name constructors and destructor */
		//@{
		/** Makes a black RGB color object.

		    \verbatim
			Some colors are already instanciated, so you can do:

				Color c;

				c = black;
				c = white;
				c = green;
				c = red;
				c = blue;
				c = yellow;
				c = cyan;
				c = pink;
				c = orange;
		    \endverbatim
		*/
		Color();

		/** Makes a color in #coding_# colorspace.

		    @param data0_ is the RED component (if #coding_# is set to "rgb") or the Y component (if #coding_# is set to "yuv")
		    @param data1_ is the GREEN component (if #coding_# is set to "rgb") or the U component (if #coding_# is set to "yuv")
		    @param data2_ is the BLUE component (if #coding_# is set to "rgb") or the V component (if #coding_# is set to "yuv")
		    @param coding_ is the coding (RGB by default)

		    \verbatim
			Example:
				Color w(255, 255, 255);
				Color r(255, 0, 0);
		    \endverbatim
		*/
		Color(unsigned char data0_, unsigned char data1_, unsigned char data2_, const char *coding_ = "rgb");

		/** Makes a RGB color from a string.

		    @param color_name is the X11 color name (see file rgb.txt for the list of color names)

		    \verbatim
			Example:
				Color w("white");
				Color r("red");
				Color p("peach puff");

			@see rgb.txt for the list of color names.
		    \endverbatim
		*/
		Color(const char *color_name_);

		/// Destructor
		~Color();
		//@}

		/**@name methods */
		//@{
		/// Set a gray scale value
		void setGray(unsigned char gray_);

		/// Set a RGB color
		void setRGB(unsigned char r_, unsigned char g_, unsigned char b_);

		/// Set a YUB color
		void setYUV(unsigned char y_, unsigned char u_, unsigned char v_);
		//@}
	};

	extern Color black;
	extern Color white;
	extern Color green;
	extern Color lightgreen;
	extern Color red;
	extern Color lightred;
	extern Color blue;
	extern Color lightblue;
	extern Color yellow;
	extern Color lightyellow;
	extern Color cyan;
	extern Color lightcyan;
	extern Color seagreen;
	extern Color pink;
	extern Color orange;

	struct Colormaker_
	{
		int red;
		int green;
		int blue;
		const char *name;
	};

	extern int Colormaker_size;
	extern struct Colormaker_ Colormaker_colors[];

}

#endif
