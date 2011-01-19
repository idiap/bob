#include "ip/ipRotate.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipRotate::ipRotate()
	:	ipCore()
{
	addIOption("centerx", 0, "Ox coordinate of the rotation center");
	addIOption("centery", 0, "Oy coordinate of the rotation center");
	addDOption("angle", 0.0, "angle in degrees of the rotation");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipRotate::~ipRotate()
{
}

//////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipRotate::checkInput(const Tensor& input) const
{
	// Accept only 3D tensors of Torch::Image type
	if (	input.nDimension() != 3 ||
		input.getDatatype() != Tensor::Short)
	{
	   	warning("ipRotate::checkInput(): Incorrect Tensor type and dimension.");
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipRotate::allocateOutput(const Tensor& input)
{
	const double degAngle = getDOption("angle");

	// Compute the size of the rotated image
	const int in_width = input.size(1);
	const int in_height = input.size(0);
	int out_width = 0;
	int out_height = 0;

	m_rotMatrix.reset(ccwAngle(degAngle));
	if (m_rotMatrix.degAngle == 0.0)
	{
		out_width = in_width;
		out_height = in_height;
	}
	else if (m_rotMatrix.degAngle == 180.0)
	{
		out_width = in_width;
		out_height = in_height;
	}
	else if (m_rotMatrix.degAngle == 90.0)
	{
		out_width = in_height;
		out_height = in_width;
	}
	else if (m_rotMatrix.degAngle == 270.0)
	{
		out_width = in_height;
		out_height = in_width;
	}
	else
	{
	   	const double dAbsSin = fabs(m_rotMatrix.dSin);
		const double dAbsCos = fabs(m_rotMatrix.dCos);

    		out_width = FixI((double)in_height * dAbsSin + (double)in_width * dAbsCos);
    		out_height = FixI((double)in_width * dAbsSin + (double)in_height * dAbsCos);
	}

	// Compute the size of the buffers to store partial results and the final rotation angle
	double angle = ccwAngle(degAngle);
	int width1 = 0;
	int height1 = 0;

		// decimate angles
	if ((angle > 45.0) && (angle <= 135.0))
    	{
        	// Angle in (45.0 .. 135.0]
	        // Rotate image by 90 degrees into temporary image,
        	// so it requires only an extra rotation angle
	        // of -45.0 .. +45.0 to complete rotation.
	   	width1 = in_height;
    		height1 = in_width;
	        angle -= 90.0;
    	}
    	else if ((angle > 135.0) && (angle <= 225.0))
    	{
        	// Angle in (135.0 .. 225.0]
	        // Rotate image by 180 degrees into temporary image,
        	// so it requires only an extra rotation angle
	        // of -45.0 .. +45.0 to complete rotation.
	   	width1 = in_width;
    		height1 = in_height;
		angle -= 180.0;
    	}
    	else if ((angle > 225.0) && (angle <= 315.0))
    	{
	        // Angle in (225.0 .. 315.0]
        	// Rotate image by 270 degrees into temporary image,
	        // so it requires only an extra rotation angle
        	// of -45.0 .. +45.0 to complete rotation.
	   	width1 = in_height;
    		height1 = in_width;
        	angle -= 270.0;
    	}
	else
	{
	   	width1 = in_width;
    		height1 = in_height;
	}

	// This is the rotation matrix of the decimated angle
	m_rotMatrix.reset(ccwAngle(angle));

	const int width2 = width1 + FixI((double)height1 * fabs(m_rotMatrix.dTan2));
    	const int height2 = height1;

	const int width3 = width2;
    	const int height3 = FixI((double)width1 * fabs(m_rotMatrix.dSin) + (double)height1 * m_rotMatrix.dCos);

    	// Allocate the rotated image
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != out_height ||
		m_output[0]->size(1) != out_width ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(out_height, out_width, input.size(2));
	}

	// Allocate temporary buffers
	m_pixmap1.resize(height1, width1, input.size(2));
	m_pixmap2.resize(height2, width2, input.size(2));
	m_pixmap3.resize(height3, width3, input.size(2));

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipRotate::processInput(const Tensor& input)
{
	const int cx = getIOption("centerx");
	const int cy = getIOption("centery");
	const double degAngle = getDOption("angle");

	// Check that the center point is within the image
	if (	cx < 0 || cx > input.size(1) ||
		cy < 0 || cy > input.size(0))
	{
	   	warning("ipRotate::processInput(): center of rotation (%d %d) out of the image (%d %d).", cx, cy, input.size(1), input.size(0));
		return false;
	}

	const int in_height = input.size(0);
	const int in_width = input.size(1);

	// Shift the input image to align the center of rotation
	m_shifter.setIOption("shiftx", in_width / 2 - cx);
	m_shifter.setIOption("shifty", in_height / 2 - cy);
	if (m_shifter.process(input) == false)
	{
		return false;
	}

	// Fill the buffer images with black
	m_pixmap1.fill(0);
	m_pixmap2.fill(0);
	m_pixmap3.fill(0);

	// Do the rotation on the shifted image
	const ShortTensor& shifted_image = (const ShortTensor&)m_shifter.getOutput(0);

	double angle = ccwAngle(degAngle);
	if (angle == 0.0)
	{
		return rotateimage0(shifted_image, (ShortTensor&)getOutput(0));
	}

	if ((angle > 0.0) && (angle <= 45.0))
	{
		return rotateimage45(shifted_image, (ShortTensor&)getOutput(0), angle);
	}

	if ((angle > 45.0) && (angle <= 135.0))
    	{
        	// Angle in (45.0 .. 135.0]
	        // Rotate image by 90 degrees into temporary image,
        	// so it requires only an extra rotation angle
	        // of -45.0 .. +45.0 to complete rotation.

		rotateimage90(shifted_image, m_pixmap1);
	        angle -= 90.0;
    	}
    	else if ((angle > 135.0) && (angle <= 225.0))
    	{
        	// Angle in (135.0 .. 225.0]
	        // Rotate image by 180 degrees into temporary image,
        	// so it requires only an extra rotation angle
	        // of -45.0 .. +45.0 to complete rotation.

		rotateimage180(shifted_image, m_pixmap1);
	        angle -= 180.0;
    	}
    	else if ((angle > 225.0) && (angle <= 315.0))
    	{
	        // Angle in (225.0 .. 315.0]
        	// Rotate image by 270 degrees into temporary image,
	        // so it requires only an extra rotation angle
        	// of -45.0 .. +45.0 to complete rotation.

		rotateimage270(shifted_image, m_pixmap1);
        	angle -= 270.0;
    	}
      	else if (angle > 315.0 )
      	{
		angle -= 360.0;
		return rotateimage45(shifted_image, (ShortTensor&)getOutput(0), angle);
	}

	if (angle == 0.0)
    	{
        	// No rotation at all
		return rotateimage0(m_pixmap1, (ShortTensor&)getOutput(0));
    	}
	else
	{
		return rotateimage45(m_pixmap1, (ShortTensor&)getOutput(0), angle);
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
//  Purpose:  Skews a row horizontally (with filtered weights)
//
//  Input:    Image to skew (+ dimensions)
//            Destination of skewed image (+ dimensions)
//            Row index
//            Skew offset
//            Relative weight of right pixel
//            Background color
//
//  Output:   None.
//
//  Remarks:  Limited to 45 degree skewing only. Filters two adjacent pixels.

void ipRotate::HorizSkew(const ShortTensor& src, ShortTensor& dst, int uRow, int iOffset, double dWeight)
{
	//const int src_height = src.size(0);
	const int src_width = src.size(1);
	//const int dst_height = dst.size(0);
	const int dst_width = dst.size(1);
	const int n_planes = src.size(2);

	const int src_stride_h = src.stride(0);
	const int src_stride_w = src.stride(1);
	const int src_stride_p = src.stride(2);
	const int dst_stride_h = dst.stride(0);
	const int dst_stride_w = dst.stride(1);
	const int dst_stride_p = dst.stride(2);

	const short* src_data = (const short*)src.dataR();
	short* dst_data = (short*)dst.dataW();

	// Do the interpolation for each color plane
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_row = &src_data[p * src_stride_p + uRow * src_stride_h];
		short* dst_row = &dst_data[p * dst_stride_p + uRow * dst_stride_h];

		// Fill gap left of skew with background
		for (int i = 0; i < iOffset; i ++, dst_row += dst_stride_w)
		{
			*dst_row = 0;
		}

		// Actual interpolation ...
		double pxlOldLeft = 0.0;
		int i = 0;
		for (i = 0; i < src_width; i ++, src_row += src_stride_w)
		{
			// Loop through row pixels
			double pxlSrc = *src_row;

			// Calculate weights
			double pxlLeft = dWeight * pxlSrc;

			// Update left over on source
			pxlSrc = pxlSrc - (pxlLeft - pxlOldLeft);

			// Check boundaries
			if ((i + iOffset >= 0) && (i + iOffset < dst_width))
			{
				*dst_row = pxlSrc;
				dst_row += dst_stride_w;
			}

			// Save leftover for next pixel in scan
			pxlOldLeft = pxlLeft;
		}

		// Go to rightmost point of skew
		i += iOffset;

		// If still in image bounds, put leftovers there
		if (i < dst_width)
		{
			*dst_row = pxlOldLeft;
			dst_row += dst_stride_w;
		}

		// Clear to the right of the skewed line with background
		while (++ i < dst_width)
		{
			*dst_row = 0;
			dst_row += dst_stride_w;
		}
	}
  dst.resetFromData();
}

/////////////////////////////////////////////////////////////////////////
//  Purpose:  Skews a column vertically (with filtered weights)
//
//  Input:    Image to skew (+dimensions)
//            Destination of skewed image (+dimensions)
//            Column index
//            Skew offset
//            Relative weight of upper pixel
//            Background color
//
//  Output:   None.
//
//  Remarks:  Limited to 45 degree skewing only. Filters two adjacent pixels.

void ipRotate::VertSkew(const ShortTensor& src, ShortTensor& dst, int uCol, int iOffset, double dWeight)
{
	const int src_height = src.size(0);
	//const int src_width = src.size(1);
	const int dst_height = dst.size(0);
	//const int dst_width = dst.size(1);
	const int n_planes = src.size(2);

	const int src_stride_h = src.stride(0);
	const int src_stride_w = src.stride(1);
	const int src_stride_p = src.stride(2);

	const int dst_stride_h = dst.stride(0);
	const int dst_stride_w = dst.stride(1);
	const int dst_stride_p = dst.stride(2);

	const short* src_data = (const short*)src.dataR();
	short* dst_data = (short*)dst.dataW();

	// Do the interpolation for each color plane
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_col = &src_data[p * src_stride_p + uCol * src_stride_w];
		short* dst_col = &dst_data[p * dst_stride_p + uCol * dst_stride_w];

		// Fill gap above skew with background
		for (int i = 0; i < iOffset; i ++, dst_col += dst_stride_h)
		{
			*dst_col = 0;
		}

		// Actual interpolation ...
		double pxlOldLeft = 0.0;
		int i = 0;
		for (i = 0; i < src_height; i ++, src_col += src_stride_h)
		{
			// Loop through column pixels
			double pxlSrc = *src_col;

			// Calculate weights
			double pxlLeft = dWeight * pxlSrc;

			// Update left over on source
			pxlSrc = pxlSrc - (pxlLeft - pxlOldLeft);

			// Check boundries
			if ((i + iOffset >= 0) && (i + iOffset < dst_height))
			{
				*dst_col = pxlSrc;
				dst_col += dst_stride_h;
			}

			// Save leftover for next pixel in scan
			pxlOldLeft = pxlLeft;
		}

		// Go to bottom point of skew
		i += iOffset;

		// If still in image bounds, put leftovers there
		if (i < dst_height)
		{
			*dst_col = pxlOldLeft;
			dst_col += dst_stride_h;
		}

		// Clear below skewed line with background
		while (++ i < dst_height)
		{
			*dst_col = 0;
			dst_col += dst_stride_h;
		}
	}
  dst.resetFromData();
}

/////////////////////////////////////////////////////////////////////////
// Special case of rotation: 0 degree

bool ipRotate::rotateimage0(const ShortTensor& src, ShortTensor& dst)
{
	dst.copy(&src);
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Special case of rotation: 90 degrees (invert height with width)

bool ipRotate::rotateimage90(const ShortTensor& src, ShortTensor& dst)
{
	const int src_height = src.size(0);
	const int src_width = src.size(1);
	const int dst_height = dst.size(0);
	//const int dst_width = dst.size(1);
	const int n_planes = src.size(2);

	const int src_stride_h = src.stride(0);
	const int src_stride_w = src.stride(1);
	const int src_stride_p = src.stride(2);

	const int dst_stride_h = dst.stride(0);
	const int dst_stride_w = dst.stride(1);
	const int dst_stride_p = dst.stride(2);

	const short* src_data = (const short*)src.dataR();
	short* dst_data = (short*)dst.dataW();

	// Destination's width/height is the source's height/width
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_plane = &src_data[p * src_stride_p];
		short* dst_plane = &dst_data[p * dst_stride_p + (dst_height - 1) * dst_stride_h];

		for (int y = 0; y < src_height; y ++)
		{
			const short* src_row = &src_plane[y * src_stride_h];
			short* dst_col = &dst_plane[y * dst_stride_w];

			for (int x = 0; x < src_width; x ++, src_row += src_stride_w, dst_col -= dst_stride_h)
			{
				*dst_col = *src_row;
			}
		}
	}
  dst.resetFromData();
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Special case of rotation: 180 degrees (mirror horizontally)

bool ipRotate::rotateimage180(const ShortTensor& src, ShortTensor& dst)
{
	const int src_height = src.size(0);
	const int src_width = src.size(1);
	const int dst_height = dst.size(0);
	const int dst_width = dst.size(1);
	const int n_planes = src.size(2);

	const int src_stride_h = src.stride(0);
	const int src_stride_w = src.stride(1);
	const int src_stride_p = src.stride(2);

	const int dst_stride_h = dst.stride(0);
	const int dst_stride_w = dst.stride(1);
	const int dst_stride_p = dst.stride(2);

	const short* src_data = (const short*)src.dataR();
	short* dst_data = (short*)dst.dataW();

	// Destination and source have the same size, but the destination is mirrored over horizontal
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_plane = &src_data[p * src_stride_p];
		short* dst_plane = &dst_data[p * dst_stride_p  + (dst_width - 1) * dst_stride_w];

		for (int y = 0; y < src_height; y ++)
		{
			const short* src_row = &src_plane[y * src_stride_h];
			short* dst_row = &dst_plane[(dst_height - y - 1) * dst_stride_h];

			for (int x = 0; x < src_width; x ++, src_row += src_stride_w, dst_row -= dst_stride_w)
			{
				*dst_row = *src_row;
			}
		}
	}
  dst.resetFromData();
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Special case of rotation: 270 degrees (invert height with width and mirror vertically)

bool ipRotate::rotateimage270(const ShortTensor& src, ShortTensor& dst)
{
    	const int src_height = src.size(0);
	const int src_width = src.size(1);
	//const int dst_height = dst.size(0);
	//const int dst_width = dst.size(1);
	const int n_planes = src.size(2);

	const int src_stride_h = src.stride(0);
	const int src_stride_w = src.stride(1);
	const int src_stride_p = src.stride(2);

	const int dst_stride_h = dst.stride(0);
	const int dst_stride_w = dst.stride(1);
	const int dst_stride_p = dst.stride(2);

	const short* src_data = (const short*)src.dataR();
	short* dst_data = (short*)dst.dataW();

	// Destination's width/height is the source's height/width
	//	and the destination is flipped over vertical
	for (int p = 0; p < n_planes; p ++)
	{
		const short* src_plane = &src_data[p * src_stride_p];
		short* dst_plane = &dst_data[p * dst_stride_p];

		for (int y = 0; y < src_height; y ++)
		{
			const short* src_row = &src_plane[(src_height - y - 1) * src_stride_h];
			short* dst_col = &dst_plane[y * dst_stride_w];

			for (int x = 0; x < src_width; x ++)
			{
				*dst_col = *src_row;
				src_row += src_stride_w;
				dst_col += dst_stride_h;
			}
		}
	}
  dst.resetFromData();
	return true;
}

/////////////////////////////////////////////////////////////////////////
//  Rotates an image by a given degree in range [-45.0 .. +45.0] (counter clockwise)
//	Using the 3-shear technique.

bool ipRotate::rotateimage45(const ShortTensor& src, ShortTensor& dst, double angle)
{
	//const int src_height = src.size(0);
	const int src_width = src.size(1);
	const int dst_height = dst.size(0);
	//const int dst_width = dst.size(1);

	// Use pixmap (2) and (3) as buffers
	const int height2 = m_pixmap2.size(0);
	//const int width2 = m_pixmap2.size(1);
	//const int height3 = m_pixmap3.size(0);
	const int width3 = m_pixmap3.size(1);

	// Perform 1st shear (horizontal)
	for (int u = 0; u < height2; u ++)
    	{
        	double dShear = m_rotMatrix.dTan2;

        	if (m_rotMatrix.dTan2 >= 0.0)
        	{
            		// Positive angle
            		dShear *= ((double)u) + 0.5;
        	}
        	else
        	{
            		// Negative angle
            		dShear *= ((double) (u - height2)) + 0.5;
        	}

        	int iShear = (int) floor(dShear);

        	HorizSkew(src, m_pixmap2, u, iShear, dShear - (double)iShear);
    	}

	// Perform 2nd shear  (vertical)
    	double dOffset;     // Variable skew offset

    	if (m_rotMatrix.dSin > 0.0)
    	{
        	// Positive angle
        	dOffset = (double)(src_width - 1) * m_rotMatrix.dSin;
    	}
    	else
    	{
        	// Negative angle
        	dOffset = -m_rotMatrix.dSin * (double)(src_width - width3);
    	}

    	for (int u = 0; u < width3; u ++, dOffset -= m_rotMatrix.dSin)
    	{
        	int iShear = (int)floor(dOffset);

        	VertSkew(m_pixmap2, m_pixmap3, u, iShear, dOffset - (double)iShear);
    	}

    	// Perform 3rd shear (horizontal)
    	if (m_rotMatrix.dSin >= 0.0)
    	{
        	// Positive angle
        	dOffset = (double)(src_width - 1) * m_rotMatrix.dSin * -m_rotMatrix.dTan2;
    	}
    	else
    	{
        	// Negative angle
        	dOffset = m_rotMatrix.dTan2 * ((double)(src_width - 1) * -m_rotMatrix.dSin + (double)(1 - dst_height));
    	}

    	for (int u = 0; u < dst_height; u ++, dOffset += m_rotMatrix.dTan2)
    	{
        	int iShear = (int)floor(dOffset);

        	HorizSkew(m_pixmap3, dst, u, iShear, dOffset - double (iShear));
    	}

	return true;
}

/////////////////////////////////////////////////////////////////////////

}
