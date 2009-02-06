#include "ipGeomNorm.h"
#include "Tensor.h"
#include "trigonometry.h"
#include "RotationMatrix2D.h"
#include "Point2D.h"
#include "Image.h"
#include "Color.h"
#include "xtprobeImageFile.h"

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
	addIOption("cropBorderX", 0, "crop: Ox border of the  normalized image around the cropping center");
	addIOption("cropBorderY", 0, "crop: Oy border of the  normalized image around the cropping center");
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
	// Check the type and size
	if (	input.getDatatype() != Tensor::Short ||
		input.nDimension() != 3)
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
	const int cropW = getIOption("cropW");
	const int cropH = getIOption("cropH");
	const int cropBorderX = getIOption("cropBorderX");
	const int cropBorderY = getIOption("cropBorderY");

	const int normW = cropW + 2 * cropBorderX;
	const int normH = cropH + 2 * cropBorderY;

	if (	m_output == 0 ||
		m_output[0]->nDimension() != 3 ||
		m_output[0]->size(0) != normH ||
		m_output[0]->size(1) != normW ||
		m_output[0]->size(2) != input.size(2))
	{
		cleanup();

		// Need allocation
		m_n_outputs = 1;
		m_output = new Tensor*[m_n_outputs];
		m_output[0] = new ShortTensor(normH, normW, input.size(2));
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated)

bool ipGeomNorm::processInput(const Tensor& input)
{
	// Get parameters
	const int rotIdx1 = getIOption("rotIdx1");
	const int rotIdx2 = getIOption("rotIdx2");
	const double rotAngle = getDOption("rotAngle");
	const int scaleIdx1 = getIOption("scaleIdx1");
	const int scaleIdx2 = getIOption("scaleIdx2");
	const int scaleDist = getIOption("scaleDist");
	const int cropIdx1 = getIOption("cropIdx1");
	const int cropIdx2 = getIOption("cropIdx2");
	const int cropDx = getIOption("cropDx");
	const int cropDy = getIOption("cropDy");
	const int cropW = getIOption("cropW");
	const int cropH = getIOption("cropH");
	const int cropBorderX = getIOption("borderX");
	const int cropBorderY = getIOption("borderY");

	const int normW = cropW + 2 * cropBorderX;
	const int normH = cropH + 2 * cropBorderY;

	// Check parameters
	if (	m_gt_pts == 0 ||
		m_n_gt_pts < 1 ||
		isIndex(rotIdx1, m_n_gt_pts) == false ||
		isIndex(rotIdx2, m_n_gt_pts) == false ||
		isIndex(scaleIdx1, m_n_gt_pts) == false ||
		isIndex(scaleIdx2, m_n_gt_pts) == false ||
		scaleIdx1 == scaleIdx2 ||
		isIndex(cropIdx1, m_n_gt_pts) == false ||
		isIndex(cropIdx2, m_n_gt_pts) == false ||
		scaleDist < 1 || scaleDist > cropW || scaleDist > cropH ||
		cropDx < -cropW || cropDx > cropW ||
		cropDy < -cropH || cropDy > cropH ||
		cropBorderX < 0 || cropBorderY < 0)
	{
		return false;
	}

	////////////////////////////////////////////////////////////
	// Rotate: the image and the ground truth points

	// Compute the rotation parameters
	const sPoint2D& rotP1 = m_gt_pts[rotIdx1];
	const sPoint2D& rotP2 = m_gt_pts[rotIdx2];
	const double rot_P12_angle = Torch::angle(Point2D(rotP1), Point2D(rotP2));

	const double ip_rot_angle = rotAngle - radian2degree(rot_P12_angle);
	const int ip_rot_cx = FixI(0.5 * (rotP1.x + rotP2.x));
	const int ip_rot_cy = FixI(0.5 * (rotP1.y + rotP2.y));

	// Rotate the image
	if (	m_ip_rotate.setDOption("angle", ip_rot_angle) == false ||
		m_ip_rotate.setIOption("centerx", ip_rot_cx) == false ||
		m_ip_rotate.setIOption("centery", ip_rot_cy) == false ||
		m_ip_rotate.process(input) == false)
	{
		return false;
	}

	// Rotate the center of the rotation -> (ip_rot_rot_cx, ip_rot_rot_cy)
	RotationMatrix2D rm;
	rm.reset(ip_rot_angle);
	rm.xc = ip_rot_cx;
	rm.yc = ip_rot_cy;
	rm.XC = m_ip_rotate.getOutput(0).size(1) / 2;
	rm.YC = m_ip_rotate.getOutput(0).size(0) / 2;

	int ip_rot_rot_cx, ip_rot_rot_cy;
	rotate(	ip_rot_cx, ip_rot_cy,
		&ip_rot_rot_cx, &ip_rot_rot_cy,
		&rm);

	// Rotate the ground truth points too
	rm.reset(ip_rot_angle);
	rm.xc = ip_rot_cx;
	rm.yc = ip_rot_cy;
	rm.XC = ip_rot_rot_cx;
	rm.YC = ip_rot_rot_cy;
	for (int i = 0; i < m_n_gt_pts; i ++)
	{
		rotate(	m_gt_pts[i].x, m_gt_pts[i].y,
			&m_nm_pts[i].x, &m_nm_pts[i].y,
			&rm);
	}

	////////////////////////////////////////////////////////////
	// Scale: the rotate image and the rotated ground truth points

	// Compute the size of the scaled image using the two reference points
	const sPoint2D& scaleP1 = m_nm_pts[scaleIdx1];
	const sPoint2D& scaleP2 = m_nm_pts[scaleIdx2];

	const double distP12 = d(Point2D(scaleP1), Point2D(scaleP2));
	const double scaleFactor = distP12 == 0.0 ? 1.0 : (scaleDist + 0.0) / distP12;

	const int scaleH = FixI(scaleFactor * m_ip_rotate.getOutput(0).size(0));
	const int scaleW = FixI(scaleFactor * m_ip_rotate.getOutput(0).size(1));

	// Scale the rotated image
	if (	m_ip_scale.setIOption("width", scaleW) == false ||
		m_ip_scale.setIOption("height", scaleH) == false ||
		m_ip_scale.process(m_ip_rotate.getOutput(0)) == false)
	{
		return false;
	}

	// Scale the rotated ground truth points
	for (int i = 0; i < m_n_gt_pts; i ++)
	{
		m_nm_pts[i].x *= scaleFactor;
		m_nm_pts[i].y *= scaleFactor;
	}

	////////////////////////////////////////////////////////////
	// Crop: the scaled & rotated image and the scaled & rotated ground truth points

	// Compute the cropping area
	const sPoint2D& cropP1 = m_nm_pts[cropIdx1];
	const sPoint2D& cropP2 = m_nm_pts[cropIdx2];

	const double cropP12x = 0.5 * (cropP1.x + cropP2.x);
	const double cropP12y = 0.5 * (cropP1.y + cropP2.y);

	const int ip_crop_x = cropP12x - (cropBorderX + cropDx);
	const int ip_crop_y = cropP12y - (cropBorderY + cropDy);
	const int ip_crop_w = normW;
	const int ip_crop_h = normH;

	if (	ip_crop_x < 0 || ip_crop_y < 0 ||
		ip_crop_w < 0 || ip_crop_h < 0 ||
		ip_crop_x + ip_crop_w > m_ip_scale.getOutput(0).size(1) ||
		ip_crop_y + ip_crop_h > m_ip_scale.getOutput(0).size(0))
	{
		return false;
	}

	// Crop the image
	if (	m_ip_crop.setIOption("x", ip_crop_x) == false ||
		m_ip_crop.setIOption("y", ip_crop_y) == false ||
		m_ip_crop.setIOption("w", ip_crop_w) == false ||
		m_ip_crop.setIOption("h", ip_crop_h) == false ||
		m_ip_crop.process(m_ip_scale.getOutput(0)) == false)
	{
		return false;
	}

	// Crop the points too
	for (int i = 0; i < m_n_gt_pts; i ++)
	{
		m_nm_pts[i].x = getInRange(FixI(m_nm_pts[i].x - ip_crop_x), 0, normW - 1);
		m_nm_pts[i].y = getInRange(FixI(m_nm_pts[i].y - ip_crop_y), 0, normH - 1);
	}

	////////////////////////////////////////////////////////////////
	// FINISH: Just copy the pixel from the ipCrop to the output

	m_output[0]->copy(&m_ip_crop.getOutput(0));

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
