#include "ipGeomNorm.h"
#include "Tensor.h"

namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipGeomNorm::ipGeomNorm()
	:	ipCore(),
		m_gt_pts(0),
		m_nm_pts(0),
		m_n_gt_pts(0)
{
	addIOption("rotIdx1", 0, "rotation: index of the first point to define the rotation axis - RP1");
	addIOption("rotIdx2", 0, "rotation: index of the second point to define the rotation axis - RP2");
	addDOption("rotAngle", 0.0, "rotation: desired angle of the RP1-RP2 in the normalized image");
	addIOption("scaleIdx1", 0, "scale: index of the first point to define the scalling factor - SP1");
	addIOption("scaleIdx2", 0, "scale: index of the second point to define the scalling factor - SP2");
	addIOption("scaleDist", 0, "scale: desired distance (SP1, SP2) in the normalized image");
	addIOption("cropIdx1", 0, "crop: index of the first point to define the cropping center - CP1");
	addIOption("cropIdx2", 0, "crop: index of the second point to define the cropping center - CP2");
	addIOption("cropDx", 0,	"crop: Ox offset of the (CP1, CP2) center in the normalized image");
	addIOption("cropDy", 0,	"crop: Oy offset of the (CP1, CP2) center in the normalized image");
	addIOption("cropW", 0, "crop: width of the base normalized image (without border)");
	addIOption("cropH", 0, "crop: height of the base normalized image (without border)");
	addIOption("cropOffsetX", 0, "crop: Ox border of the  normalized image around the cropping center");
	addIOption("cropOffsetY", 0, "crop: Oy border of the  normalized image around the cropping center");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipGeomNorm::~ipGeomNorm()
{
	delete[] m_nm_pts;
}

/////////////////////////////////////////////////////////////////////////
// Change the ground truth points to use for normalization

bool ipGeomNorm::setGTPoints(const sPoint2D* gt_pts, int n_gt_pts)
{
	// Check parameters
	if (n_gt_pts < 1 || gt_pts == 0)
	{
		return false;
	}

	// OK
	m_gt_pts = gt_pts;
	delete[] m_nm_pts;
	m_nm_pts = new sPoint2D[n_gt_pts];
	m_n_gt_pts = n_gt_pts;
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type

bool ipGeomNorm::checkInput(const Tensor& input) const
{
	// Only requirement: at least 2D input tensor
	if (input.nDimension() < 2)
	{
		return false;
	}

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions

bool ipGeomNorm::allocateOutput(const Tensor& input)
{
	/*
	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != input.size(0) ||
		m_output[0]->size(1) != input.size(1) ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(input.size(0), input.size(1), input.size(2));
		return true;
	}
	*/

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipGeomNorm::processInput(const Tensor& input)
{
	/*
	// Prepare pointers to access pixels
	const ShortTensor* t_input = (ShortTensor*)&input;
	ShortTensor* t_output = (ShortTensor*)m_output[0];

	const short* src = (const short*)t_input->dataR();
	short* dst = (short*)t_output->dataW();

	const int stride_h = t_input->t->stride[0];	// height
	const int stride_w = t_input->t->stride[1];	// width
	const int stride_p = t_input->t->stride[2];	// no planes

	// An index for the 3D tensor is: [y * stride_h + x * stride_w + p * stride_p]
	const int width = input.size(1);
	const int height = input.size(0);
	const int n_planes = input.size(2);

	// Fill the result image with black
	t_output->fill(0);

	// Compute the range of valid pixel positions in the shifted image
	const int start_x = getInRange(m_dx, 0, width - 1);
	const int start_y = getInRange(m_dy, 0, height - 1);
	const int stop_x = getInRange(width + m_dx, 0, width - 1);
	const int stop_y = getInRange(height + m_dy, 0, height - 1);
	const int dindex = m_dy * stride_h + m_dx * stride_w;

	// Shift each plane ...
	for (int p = 0; p < n_planes; p ++)
	{
		//	input: 	[y * stride_h + x * stride_w + p * stride_p]
		//		->>>
		//	output: [(y + dy) * stride_h + (x + dx) * stride_w + p * stride_p])
		const short* src_plane = &src[p * stride_p];
		short* dst_plane = &dst[p * stride_p];

		for (int y = start_y; y < stop_y; y ++)
		{
			const int index_row = y * stride_h + start_x * stride_w;
			const short* src_row = &src_plane[index_row - dindex];
			short* dst_row = &dst_plane[index_row];

			for (int x = start_x; x < stop_x; x ++, src_row += stride_w, dst_row += stride_w)
			{
				*dst_row = *src_row;
			}
		}
	}
	*/

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
