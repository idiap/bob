#ifndef CONVERT_INC
#define CONVERT_INC

#include "core/general.h"

namespace Torch {

// Macros for RGB, YUV encoding/decoding

#define FIXNUM 16
#define FIX(a, b) ((int)((a)*(1<<(b))))
#define UNFIX(a, b) ((a+(1<<(b-1)))>>(b))

// Approximate 255 by 256
#define ICCIRUV(x) (((x)<<8)/224)
#define ICCIRY(x) ((((x)-16)<<8)/219)

/** Clip out-range values

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
#define CLIP(t) (((t)>255)?255:(((t)<0)?0:(t)))

/** Getting R value from Y U V

    @version 2.0
    \date
    @since 1.0
*/
#define GET_R_FROM_YUV(y, u, v) UNFIX((FIX(1.0, FIXNUM)*(y) + FIX(1.402, FIXNUM)*(v)), FIXNUM)
   
/** Getting G value from Y U V

    @version 2.0
    \date
    @since 1.0
*/
#define GET_G_FROM_YUV(y, u, v) UNFIX((FIX(1.0, FIXNUM)*(y) + FIX(-0.344, FIXNUM)*(u) + FIX(-0.714, FIXNUM)*(v)), FIXNUM)

/** Getting B value from Y U V

    @version 2.0
    \date
    @since 1.0
*/
#define GET_B_FROM_YUV(y, u, v) UNFIX((FIX(1.0, FIXNUM)*(y) + FIX(1.772, FIXNUM)*(u)), FIXNUM)

/** Getting Y value from R G B

    @version 2.0
    \date
    @since 1.0
*/
#define GET_Y_FROM_RGB(r, g, b) UNFIX((FIX(0.299, FIXNUM)*(r) + FIX(0.587, FIXNUM)*(g) + FIX(0.114, FIXNUM)*(b)), FIXNUM)
   
/** Getting U value from R G B

    @version 2.0
    \date
    @since 1.0
*/
#define GET_U_FROM_RGB(r, g, b) UNFIX((FIX(-0.169, FIXNUM)*(r) + FIX(-0.331, FIXNUM)*(g) + FIX(0.500, FIXNUM)*(b)), FIXNUM)
   
/** Getting V value from R G B

    @version 2.0
    \date
    @since 1.0
*/
#define GET_V_FROM_RGB(r, g, b) UNFIX((FIX(0.500, FIXNUM)*(r) + FIX(-0.419, FIXNUM)*(g) + FIX(-0.081, FIXNUM)*(b)), FIXNUM)

/** RGB <-> YUV conversions

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
//@{

/** Pixel convertion from RGB to gray (in bytes)

    @param r_ red value (in bytes)
    @param g_ green value (in bytes)
    @param b_ blue value (in bytes)
    @return gray value (in bytes)

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
unsigned char rgb_to_gray(unsigned char r_, unsigned char g_, unsigned char b_);

/// Pixmap convertion from RGB24 to RGB24 (in bytes)
void rgb24_to_rgb24(unsigned char *data_in_, unsigned char *data_out_, unsigned int width_, unsigned int height_);

/// Pixmap convertion from RGB24 to BGR24 (in bytes)
void rgb24_to_bgr24_flip(unsigned char *data_in_, unsigned char *data_out_, unsigned int width_, unsigned int height_);

/// Pixel convertion from YUV to RGB (in bytes)
void yuv_to_rgb(unsigned char y_, unsigned char u_, unsigned char v_, unsigned char *r_, unsigned char *g_, unsigned char *b_);

/// Pixmap convertion from YUV to RGB24 (in bytes)
void yuv_to_rgb24(unsigned char *data_in_, unsigned char *data_out_, unsigned int width_, unsigned int height_);

/// Pixmap convertion from YUV422 to RGB24 (in bytes)
void yuv422_to_rgb24(unsigned char *data_in_, unsigned char *data_out_, unsigned int width_, unsigned int height_);

/// Pixmap convertion from YUV422P to YUV (in bytes) with bytes-per-line argument
bool YUV422P_to_YUV(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from YUV422P to RGB24 (in bytes) with bytes-per-line argument
bool YUV422P_to_RGB24(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from YUV420P to RGB24 (in bytes) with bytes-per-line argument
bool YUV420P_to_RGB24(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from YUV420P to GRAY (in bytes) with bytes-per-line argument
bool YUV420P_to_GREY(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from YUV420P to YUV (in bytes) with bytes-per-line argument
bool YUV420P_to_YUV(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from RGB24 to RGB24 (in bytes) with bytes-per-line argument
bool RGB24_to_RGB24(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from RGB32 to RGB24 (in bytes) with bytes-per-line argument
bool RGB32_to_RGB24(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

/// Pixmap convertion from RGB32 to YUV (in bytes) with bytes-per-line argument
bool RGB32_to_YUV(int width_, int height_, unsigned char *s_, int bytesperline_src_, unsigned char *d_, int bytesperline_dst_);

//@}

/** Y Cb Cr -> RGB conversion

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 2.0
*/
//@{
void ycbcr2rgb(int y, int cb, int cr, unsigned char *rgb);
//@}

/** Look-up tables for fast RGB -> luminance calculation

    \verbatim
	Example:
		gray = ((times77[r] + times150[g] + times29[b]) >> 8);
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
//@{

#define CONVERT_FAST_RGB2GRAY_

/// Look-up table for R
extern int times77[256];
/// Look-up table for G
extern int times150[256];
/// Look-up table for B
extern int times29[256];

//@}

/** Look-up tables for fast YUV to ICCIRYUV convertion

    \verbatim
	Example:
		Y = LUT_ICCIRY[yy];
		U = LUT_ICCIRUV_128[uu];
		V = LUT_ICCIRUV_128[vv];
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
//@{

#define FAST_ICCIRYUV

/// Look-up table for Y to ICCIRY convertion
extern int LUT_ICCIRY[256];
/// Look-up table for UV to ICCIRUV convertion
extern int LUT_ICCIRUV_128[256];

//@}

/** Look-up tables for fast ICCIRYUV to RGB convertion 

    These tables should be initialized first by calling #initLUT_ICCIRYUV_RGB()#.

    \verbatim
	Example:
		initLUT_ICCIRYUV_RGB();
		
		r = LUT_ICCIRYUV_R[yy][uu][vv];
		g = LUT_ICCIRYUV_G[yy][uu][vv];
		b = LUT_ICCIRYUV_B[yy][uu][vv];
    \endverbatim

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \date
    @since 1.0
*/
//@{

//#define FAST_ICCIRYUV_RGB

/// Look-up table for ICCIR YUV to R convertion
extern unsigned char LUT_ICCIRYUV_R[256][256][256];
/// Look-up table for ICCIR YUV to G convertion
extern unsigned char LUT_ICCIRYUV_G[256][256][256];
/// Look-up table for ICCIR YUV to B convertion
extern unsigned char LUT_ICCIRYUV_B[256][256][256];
/// Initialize the look-up tables #LUT_ICCIRYUV_R#, #LUT_ICCIRYUV_G# and #LUT_ICCIRYUV_B#
void initLUT_ICCIRYUV_RGB();

//@}

}

#endif
