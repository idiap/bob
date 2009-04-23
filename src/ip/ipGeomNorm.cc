#include "ipGeomNorm.h"
#include "Tensor.h"
#include "trigonometry.h"
#include "RotationMatrix2D.h"
#include "Point2D.h"
#include "Image.h"
#include "Color.h"
#include "xtprobeImageFile.h"
#include "GTFile.h"


namespace Torch {

/////////////////////////////////////////////////////////////////////////
// Constructor

ipGeomNorm::ipGeomNorm()
	:	ipCore(),
		m_gt_file(0),
		m_nm_pts(0)
{
	addSOption("rotPoint1", "", "rotation: first point to define the rotation axis - RP1");
	addSOption("rotPoint2", "", "rotation: second point to define the rotation axis - RP2");
	addDOption("rotAngle", 0.0, "rotation: desired angle of the RP1-RP2 in the normalized image");
	addSOption("scalePoint1", "", "scale: first point to define the scalling factor - SP1");
	addSOption("scalePoint2", "", "scale: second point to define the scalling factor - SP2");
	addIOption("scaleDist", 0, "scale: desired distance (SP1, SP2) in the normalized image");
	addSOption("cropPoint1", "", "crop: first point to define the cropping center - CP1");
	addSOption("cropPoint2", "", "crop: second point to define the cropping center - CP2");
	addIOption("cropDx", 0,	"crop: Ox offset of the (CP1, CP2) center in the normalized image");
	addIOption("cropDy", 0,	"crop: Oy offset of the (CP1, CP2) center in the normalized image");
	addIOption("cropW", 0, "crop: width of the base normalized image (without border)");
	addIOption("cropH", 0, "crop: height of the base normalized image (without border)");
	addIOption("cropBorderX", 0, "crop: Ox border of the  normalized image around the cropping center");
	addIOption("cropBorderY", 0, "crop: Oy border of the  normalized image around the cropping center");
	addDOption("finalRotAngle", 0.0, "final rotation: final rotation angle of the center");

	addBOption("verbose", false, "verbose");
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipGeomNorm::~ipGeomNorm()
{
	delete[] m_nm_pts;
}

/////////////////////////////////////////////////////////////////////////
// Load the configuration parameters from a text file

const char* normString(const char* str)
{
	const char* ret = str;
	while (*ret != '\0' && *ret != ':')
	{
		ret ++;
	}
	if (*ret == ':')
	{
		ret ++;
	}
	return ret;
}

bool ipGeomNorm::loadCfg(const char* filename)
{
	const bool verbose = getBOption("verbose");

	// Open the file
	File file;
	if (file.open(filename, "r") == false)
	{
		return false;
	}

	static const int sizeBuf = 512;
	char str[sizeBuf];

	// All parameters
	char rotPoint1[sizeBuf];
	char rotPoint2[sizeBuf];
	double rotAngle;
	char scalePoint1[sizeBuf];
	char scalePoint2[sizeBuf];
	int scaleDist;
	char cropPoint1[sizeBuf];
	char cropPoint2[sizeBuf];
	int cropDx, cropDy, cropW, cropH, cropBorderX, cropBorderY;
	double finalRotAngle;

	// Read the rotation parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%s %s %lf", rotPoint1, rotPoint2, &rotAngle) != 3)
	{
		if (verbose == true)
		{
			warning("ipGeomNorm::loadCfg - Error at [%s] - [%s]!\n", str, normString(str));
		}
		return false;
	}

	// Read the scalling parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%s %s %d", scalePoint1, scalePoint2, &scaleDist) != 3)
	{
		if (verbose == true)
		{
			warning("ipGeomNorm::loadCfg - Error at [%s] - [%s]!\n", str, normString(str));
		}
		return false;
	}

	// Read the cropping parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%s %s %d %d %d %d %d %d",
			cropPoint1, cropPoint2,
			&cropDx, &cropDy, &cropW, &cropH, &cropBorderX, &cropBorderY) != 8)
	{
		if (verbose == true)
		{
			warning("ipGeomNorm::loadCfg - Error at [%s] - [%s]!\n", str, normString(str));
		}
		return false;
	}

	// Read the final rotation angle
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%lf", &finalRotAngle) != 1)
	{
		if (verbose == true)
		{
			warning("ipGeomNorm::loadCfg - Error at [%s] - [%s]!\n", str, normString(str));
		}
		return false;
	}

	file.close();

    // Verbose
	if (verbose == true)
	{
		print("\nRotation: pt1 = %s, pt2 = %s, angle = %lf\n",
			rotPoint1, rotPoint2, rotAngle);
		print("Scalling: pt1 = %s, pt2 = %s, distance = %d\n",
			scalePoint1, scalePoint1, scaleDist);
		print("Cropping: pt1 = %s, pt2 = %s, dx = %d, dy = %d, w = %d, h = %d, borderx = %d, bordery = %d\n",
			cropPoint1, cropPoint2, cropDx, cropDy, cropW, cropH, cropBorderX, cropBorderY);
		print("Final rotation: angle = %lf\n",
			finalRotAngle);
	}

	// OK, set parameters
	return	setSOption("rotPoint1", rotPoint1) == true &&
		setSOption("rotPoint2", rotPoint2) == true &&
		setDOption("rotAngle", rotAngle) == true &&
		setSOption("scalePoint1", scalePoint1) == true &&
		setSOption("scalePoint2", scalePoint2) == true &&
		setIOption("scaleDist", scaleDist) == true &&
		setSOption("cropPoint1", cropPoint1) == true &&
		setSOption("cropPoint2", cropPoint2) == true &&
		setIOption("cropDx", cropDx) == true &&
		setIOption("cropDy", cropDy) == true &&
		setIOption("cropW", cropW) == true &&
		setIOption("cropH", cropH) == true &&
		setIOption("cropBorderX", cropBorderX) == true &&
		setIOption("cropBorderY", cropBorderY) == true &&
		setDOption("finalRotAngle", finalRotAngle);
}

/////////////////////////////////////////////////////////////////////////
// Change the ground truth points to use for normalization

bool ipGeomNorm::setGTFile(const GTFile* gt_file)
{
	// Check parameters
	if (gt_file == 0 || gt_file->getNPoints() < 1)
	{
		return false;
	}

	// OK
	m_gt_file = gt_file;
	delete[] m_nm_pts;
	m_nm_pts = new sPoint2D[m_gt_file->getNPoints()];
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
	   	warning("ipGeomNorm::checkInput(): Incorrect Tensor type and dimension.");
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

        if (    m_output == 0 ||
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
	//const bool verbose = getBOption("verbose");

	// Get parameters
	const char* rotPoint1 = getSOption("rotPoint1");
	const char* rotPoint2 = getSOption("rotPoint2");
	const double rotAngle = getDOption("rotAngle");
	const char* scalePoint1 = getSOption("scalePoint1");
	const char* scalePoint2 = getSOption("scalePoint2");
	const int scaleDist = getIOption("scaleDist");
	const char* cropPoint1 = getSOption("cropPoint1");
	const char* cropPoint2 = getSOption("cropPoint2");
	const int cropDx = getIOption("cropDx");
	const int cropDy = getIOption("cropDy");
	const int cropW = getIOption("cropW");
	const int cropH = getIOption("cropH");
	const int cropBorderX = getIOption("cropBorderX");
	const int cropBorderY = getIOption("cropBorderY");
	const double finalRotAngle = getDOption("finalRotAngle");

	const int normW = cropW + 2 * cropBorderX;

	const int normH = cropH + 2 * cropBorderY;


	// Check parameters
	if (	m_gt_file == 0 ||
		m_gt_file->getNPoints() < 1 ||
		m_gt_file->hasLabel(rotPoint1) == false ||
		m_gt_file->hasLabel(rotPoint2) == false ||
		m_gt_file->hasLabel(scalePoint1) == false ||
		m_gt_file->hasLabel(scalePoint2) == false ||
		strcmp(scalePoint1, scalePoint2) == 0 ||
		m_gt_file->hasLabel(cropPoint1) == false ||
		m_gt_file->hasLabel(cropPoint2) == false ||
		scaleDist < 1 || scaleDist > cropW || scaleDist > cropH ||
		cropDx < -cropW || cropDx > cropW ||
		cropDy < -cropH || cropDy > cropH ||
		cropBorderX < 0 || cropBorderY < 0)
	{
		warning("ipGeomNorm::processInput(): incorrect parameters.");
		return false;
	}

	////////////////////////////////////////////////////////////
	// Transform the labels to indexes

	const sPoint2D* gt_pts = m_gt_file->getPoints();
	const int n_gt_pts = m_gt_file->getNPoints();

	const int rotIdx1 = m_gt_file->getIndex(rotPoint1);
	const int rotIdx2 = m_gt_file->getIndex(rotPoint2);

	const int cropIdx1 = m_gt_file->getIndex(cropPoint1);
	const int cropIdx2 = m_gt_file->getIndex(cropPoint2);

	const int scaleIdx1 = m_gt_file->getIndex(scalePoint1);
	const int scaleIdx2 = m_gt_file->getIndex(scalePoint2);

	////////////////////////////////////////////////////////////
	// Rotate: the image and the ground truth points

	// Compute the rotation parameters
	const sPoint2D& rotP1 = gt_pts[rotIdx1];
	const sPoint2D& rotP2 = gt_pts[rotIdx2];
	double rot_P12_angle = Torch::angle(Point2D(rotP1), Point2D(rotP2));


	if(gt_pts[rotIdx1].y < gt_pts[rotIdx2].y)
		rot_P12_angle = -rot_P12_angle;

        double ip_rot_angle = -rotAngle + radian2degree(rot_P12_angle);

	//
	double v_rot_cx = (rotP1.x + rotP2.x)/2;
	double v_rot_cy = (rotP1.y + rotP2.y)/2;

	//rotate the point just to find the center
	RotationMatrix2D rm;
	rm.reset(ip_rot_angle);
	rm.xc = v_rot_cx;
	rm.yc = v_rot_cy;
	rm.XC = v_rot_cx;
	rm.YC = v_rot_cy;

	// Compute the size of the scaled image using the two reference points
	const sPoint2D& scaleP1 =gt_pts[scaleIdx1];
	const sPoint2D& scaleP2 = gt_pts[scaleIdx2];

	const double distP12 = d(Point2D(scaleP1), Point2D(scaleP2));
	const double scaleFactor = distP12 == 0.0 ? 1.0 :   distP12/(scaleDist + 0.0) ;
	// print("Scaledistance %d\n",scaleDist);
	double px1,px2,py1,py2;
	rotate(	gt_pts[cropIdx1].x, gt_pts[cropIdx1].y,
		&px1, &py1,
		&rm);
	rotate(	gt_pts[cropIdx2].x, gt_pts[cropIdx2].y,
		&px2, &py2,
		&rm);
	//now you have rotated points of those two points which are used for croping
	// find the center now in rotated coordinate.
	//if(verbose) print("The y coordinate %f %f, %f, %f, angle %f\n",gt_pts[cropIdx1].y, gt_pts[cropIdx2].y,py1,py2, ip_rot_angle);
	double Cx,Cy;
	Cx = (px1+px2 + (cropW-2*cropDx)*scaleFactor)/2;
	Cy = (py1+py2 + (cropH-2*cropDy)*scaleFactor)/2;

	rm.reset(-ip_rot_angle);
	rm.xc = v_rot_cx;
	rm.yc = v_rot_cy;
	rm.XC = v_rot_cx;
	rm.YC = v_rot_cy;
	//now we obtain the centers in original image.
	rotate(	Cx, Cy,
		&px1, &py1,
		&rm);

	int nCx = (int)( px1+0.5);
	int nCy = (int) (py1+0.5);
	print("The centers %d %d\n",nCx,nCy);
	ip_rot_angle = finalRotAngle-ip_rot_angle;

	// Rotate the image
	if (	m_ip_rotate.setDOption("angle", ip_rot_angle) == false ||
		m_ip_rotate.setIOption("centerx", nCx) == false ||
		m_ip_rotate.setIOption("centery", nCy) == false ||
		m_ip_rotate.process(input) == false)
	{
	   	warning("ipGeomNorm::processInput(): incorrect rotation parameters or rotation failure.");
		return false;
	}

	//rotate all the points
	rm.reset(ip_rot_angle);
	rm.xc = px1;
	rm.yc = py1;
	rm.XC = m_ip_rotate.getOutput(0).size(1)/2;
	rm.YC = m_ip_rotate.getOutput(0).size(0)/2;
	for (int i = 0; i < n_gt_pts; i ++)
	{
		rotate(	gt_pts[i].x, gt_pts[i].y,
			&m_nm_pts[i].x, &m_nm_pts[i].y,
			&rm);
	}
	const int scaleH = FixI(m_ip_rotate.getOutput(0).size(0)/scaleFactor);
	const int scaleW = FixI(m_ip_rotate.getOutput(0).size(1)/scaleFactor);
	// print("Scale  = %f %d %d\n",scaleFactor,scaleH,scaleW);
	// Scale the rotated image
	if (	m_ip_scale.setIOption("width", scaleW) == false ||
		m_ip_scale.setIOption("height", scaleH) == false ||
		m_ip_scale.process(m_ip_rotate.getOutput(0)) == false)
	{
	   	warning("ipGeomNorm::processInput(): incorrect scaling parameters or scaling failure.");
		return false;
	}

	for (int i = 0; i < n_gt_pts; i ++)
	{
            m_nm_pts[i].x = m_nm_pts[i].x/scaleFactor;
             m_nm_pts[i].y = m_nm_pts[i].y/scaleFactor;

	}
	////////////////////////////////////////////////////////////
	// Crop: the scaled & rotated image and the scaled & rotated ground truth points

	// Compute the cropping area
	const int ip_crop_x =  FixI((m_ip_scale.getOutput(0).size(1) - cropW)/2)-cropBorderX;
	const int ip_crop_y = FixI((m_ip_scale.getOutput(0).size(0) - cropH)/2)-cropBorderY;
	const int ip_crop_w = normW;
	const int ip_crop_h = normH;

	if (	ip_crop_x < 0 || ip_crop_y < 0 ||
		ip_crop_w < 0 || ip_crop_h < 0 ||
		ip_crop_x + ip_crop_w > m_ip_scale.getOutput(0).size(1) ||
		ip_crop_y + ip_crop_h > m_ip_scale.getOutput(0).size(0))
	{
	   	warning("ipGeomNorm::processInput(): incorrect crop parameters.");
		return false;
	}


	// Crop the image
	if (	m_ip_crop.setIOption("x", ip_crop_x) == false ||
		m_ip_crop.setIOption("y", ip_crop_y) == false ||
		m_ip_crop.setIOption("w", ip_crop_w) == false ||
		m_ip_crop.setIOption("h", ip_crop_h) == false ||
		m_ip_crop.process(m_ip_scale.getOutput(0)) == false)
	{
	   	warning("ipGeomNorm::processInput(): crop failure.");
		return false;
	}

	// Crop the points too
	for (int i = 0; i < n_gt_pts; i ++)
	{
		m_nm_pts[i].x = getInRange(FixI(m_nm_pts[i].x - ip_crop_x), 0, normW - 1);
		m_nm_pts[i].y = getInRange(FixI(m_nm_pts[i].y - ip_crop_y), 0, normH - 1);
	}

	////////////////////////////////////////////////////////////////
	// FINISH: Just copy the pixel from the cropped image


	m_output[0]->copy(&m_ip_crop.getOutput(0));

	// OK
	return true;
}


/////////////////////////////////////////////////////////////////////////

}
