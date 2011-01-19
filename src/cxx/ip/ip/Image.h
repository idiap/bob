#ifndef IMAGE_INC
#define IMAGE_INC

#include "core/Tensor.h"

namespace Torch {

	class ImageFile;
	class Color;
	struct sRect2D;

	/** This class is designed to handle images.
		An image is stored as a 3D tensor of short with the dimensions:
			height x
			width x
			number of color components (no of planes)

	    @author Sebastien Marcel (marcel@idiap.ch)
	    @version 2.0
	    \date
	    @since 1.0
	*/
	class Image : public ShortTensor
	{
	public:

		/**@name Constructor
			Creates a black image.
			@param width is the width of the image
			@param height is the height of the image
			@param n_planes is the number of color components of the image (3 for RGB, 1 for gray)
		*/
		Image(int width = 1, int height = 1, int n_planes = 1);

		/// Destructor
		~Image();

		/// Resize the image (to new dimensions, no of planes and storage type)
		bool			resize(int width, int height, int n_planes);

		/// Copy from some 3D tensor (should have the same dimension)
		///	- all planes are copied
		bool			copyFrom(const Tensor& data);

		/// Copy from another image (should have the same dimension)
		///	- all planes are copied
		bool			copyFrom(const Image& image);

		/// Fills a pixmap from an image object
		static  void		fillPixmap(	unsigned char* pixmap, int n_planes_pixmap,
							const Image& image);

		/// Fills an image object from a pixmap (the pixmap is RGB, the image may be gray)
		static void		fillImage(	const unsigned char* pixmap, int n_planes_pixmap,
							Image& image);

		/**@name drawing methods */
		//@{
		/** draw a pixel in the image.

		    @param x is the #x# coordinate of the pixel to draw
		    @param y is the #y# coordinate of the pixel to draw
		    @param color is the color to draw
		*/
		void			drawPixel(int x, int y, const Color& color);

		/** draw a line P1-P2 in the image.

		    @param x1 is the #x# coordinate of #P1#
		    @param y1 is the #y# coordinate of #P1#
		    @param x2 is the #x# coordinate of #P2#
		    @param y2 is the #y# coordinate of #P2#
		    @param color is the color to draw
		*/
		void			drawLine(int x1, int y1, int x2, int y2, const Color& color);

		/** draw a cross of some radius centered in the given point P.

		    @param x is the #x# coordinate of #P#
		    @param y is the #y# coordinate of #P#
		    @param r is the radius
		    @param color is the color to draw
		*/
		void			drawCross(int x, int y, int r, const Color& color);

		/** draw a rectangle in the image.

		    @param color is the color to draw
		*/
		void			drawRect(int x, int y, int w, int h, const Color& color);
		void			drawRect(const sRect2D& rect, const Color& color);
		//@}

		/////////////////////////////////////////////////////////////////////////////////////////
		/// Access functions

		int			getWidth() const;
		int			getHeight() const;
		int			getNPlanes() const;

	private:

		/////////////////////////////////////////////////////////////////////////////////////

		/// Delete the allocate memory
		void			cleanup();

		/// Various functions for changing some pixel for 1D/3D images
		static void		setPixel1DChar(ShortTensor* data, int x, int y, const Color& color);
		static void		setPixel3DChar(ShortTensor* data, int x, int y, const Color& color);

		/////////////////////////////////////////////////////////////////////////////////////
		/// Attributes

		typedef void (* FuncSetPixel)(ShortTensor* data, int x, int y, const Color& color);

		FuncSetPixel		m_setPixelCallback;	// Fast callback to change some pixel's value
	};
}

#endif
