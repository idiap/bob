#include "ipGeomNorm.h"

#include "eyecenterGTFile.h"
#include "bancaGTFile.h"
#include "cootesGTFile.h"

#include "Image.h"
#include "xtprobeImageFile.h"
#include "CmdLine.h"
#include "Color.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Geometric normalization parameters (check ipGeomNorm's options)
///////////////////////////////////////////////////////////////////////////

struct GeomNormParams
{
	// Constructor
	GeomNormParams()
		:	m_rotIdx1(0), m_rotIdx2(0), m_rotAngle(0.0),
			m_scaleIdx1(0), m_scaleIdx2(0), m_scaleDist(0),
			m_cropIdx1(0), m_cropIdx2(0), m_cropDx(0), m_cropDy(0),
			m_cropW(0), m_cropH(0), m_cropBorderX(0), m_cropBorderY(0)
	{
	}

	// Set the parameters as options to some ipGeomNorm
	bool		set(ipGeomNorm& gnormalizer)
	{
		CHECK_ERROR(gnormalizer.setIOption("rotIdx1", m_rotIdx1));
		CHECK_ERROR(gnormalizer.setIOption("rotIdx2", m_rotIdx2));
		CHECK_ERROR(gnormalizer.setDOption("rotAngle", m_rotAngle));
		CHECK_ERROR(gnormalizer.setIOption("scaleIdx1", m_scaleIdx1));
		CHECK_ERROR(gnormalizer.setIOption("scaleIdx2", m_scaleIdx2));
		CHECK_ERROR(gnormalizer.setIOption("scaleDist", m_scaleDist));
		CHECK_ERROR(gnormalizer.setIOption("cropIdx1", m_cropIdx1));
		CHECK_ERROR(gnormalizer.setIOption("cropIdx2", m_cropIdx2));
		CHECK_ERROR(gnormalizer.setIOption("cropDx", m_cropDx));
		CHECK_ERROR(gnormalizer.setIOption("cropDy", m_cropDy));
		CHECK_ERROR(gnormalizer.setIOption("cropW", m_cropW));
		CHECK_ERROR(gnormalizer.setIOption("cropH", m_cropH));
		CHECK_ERROR(gnormalizer.setIOption("cropBorderX", m_cropBorderX));
		CHECK_ERROR(gnormalizer.setIOption("cropBorderY", m_cropBorderY));

		return true;
	}

	///////////////////////////////////////////////////////////////////
	// Attributes

	// Rotation
	int 		m_rotIdx1, m_rotIdx2;
	double		m_rotAngle;

	// Scalling
	int		m_scaleIdx1, m_scaleIdx2;
	int		m_scaleDist;

	// Cropping
	int		m_cropIdx1, m_cropIdx2;
	int		m_cropDx, m_cropDy;
	int		m_cropW, m_cropH;
	int		m_cropBorderX, m_cropBorderY;
};

///////////////////////////////////////////////////////////////////////////
// Parse a geometrical normalization configuration file
///////////////////////////////////////////////////////////////////////////

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

bool loadGeomNormCfg(const char* filename, GeomNormParams& params)
{
	// Open the file
	File file;
	if (file.open(filename, "r") == false)
	{
		return false;
	}

	const int sizeBuf = 512;
	char str[sizeBuf];

	// Read the rotation parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%d %d %lf",
			&params.m_rotIdx1, &params.m_rotIdx2, &params.m_rotAngle) != 3)
	{
		print("[%s] - [%s]\n", str, normString(str));
		return false;
	}

	// Read the scalling parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%d %d %d",
			&params.m_scaleIdx1, &params.m_scaleIdx2, &params.m_scaleDist) != 3)
	{
		print("[%s] - [%s]\n", str, normString(str));
		return false;
	}

	// Read the cropping parameters
	if (	file.gets(str, sizeBuf) == 0 ||
		sscanf(normString(str), "%d %d %d %d %d %d %d %d",
			&params.m_cropIdx1, &params.m_cropIdx2,
			&params.m_cropDx, &params.m_cropDy,
			&params.m_cropW, &params.m_cropH,
			&params.m_cropBorderX, &params.m_cropBorderY) != 8)
	{
		print("[%s] - [%s]\n", str, normString(str));
		return false;
	}

	file.close();

	// Verbose
	print("\nRotation: pt1 = %d, pt2 = %d, angle = %lf\n",
		params.m_rotIdx1, params.m_rotIdx2, params.m_rotAngle);
	print("Scalling: pt1 = %d, pt2 = %d, distance = %d\n",
		params.m_scaleIdx1, params.m_scaleIdx2, params.m_scaleDist);
	print("Cropping: pt1 = %d, pt2 = %d, dx = %d, dy = %d, w = %d, h = %d, borderx = %d, bordery = %d\n",
		params.m_cropIdx1, params.m_cropIdx2, params.m_cropDx, params.m_cropDy,
		params.m_cropW, params.m_cropH, params.m_cropBorderX, params.m_cropBorderY);

	// OK
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Save an image and draw the ground truth on top
///////////////////////////////////////////////////////////////////////////

void saveImageGTPts(	const ShortTensor& timage, const sPoint2D* gt_pts, int n_gt_pts,
			const char* filename)
{
	const int h = timage.size(0);
	const int w = timage.size(1);

	Image image(w, h, timage.size(2));
	image.copyFrom(timage);
	for (int i = 0; i < n_gt_pts; i ++)
	{
		const int x = gt_pts[i].x;
		const int y = gt_pts[i].y;
		if (	x > 4 && x + 4 < w &&
			y > 4 && y + 4 < h)
		{
			image.drawLine(x - 4, y, x + 4, y, Torch::red);
			image.drawLine(x, y - 4, x, y + 4, Torch::red);
		}
	}

	xtprobeImageFile xtprobe;
	xtprobe.save(image, filename);
}

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* image_filename = 0;
	char* gt_pts_filename = 0;
	char* cfg_norm_filename = 0;
	char* norm_image_filename = 0;
	int gt_format;
	bool verbose;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for geometric image normalization.\n");
	cmd.addSCmdArg("image", &image_filename, "input image file");
	cmd.addSCmdArg("ground truth positions", &gt_pts_filename, "input ground truth positions");
	cmd.addSCmdArg("configuration", &cfg_norm_filename, "input normalization configuration file");
	cmd.addSCmdArg("normalized image", &norm_image_filename, "normalized image file");
	cmd.addText("\nOptions:");
	cmd.addICmdOption("-gt_format", &gt_format, 1, "gt format (1=eyes center, 2=banca format, 3=eyes corners, 4=eye corners + nose tip + chin, 5=left eye corners + right eye center + nose tip + chin, 6=left eye center + nose tip + chin, 7=Tim Cootes's markup 68 pts)");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	cmd.read(argc, argv);

	ipGeomNorm gnormalizer;

	///////////////////////////////////////////////////////////////////
	// Load the image

	Image image(1, 1, 3);
	xtprobeImageFile xtprobe;
	CHECK_FATAL(xtprobe.load(image, image_filename) == true);

	print("Processing image [width = %d, height = %d, nplanes = %d] ...\n",
		image.size(1), image.size(0), image.size(2));

	///////////////////////////////////////////////////////////////////
	// Load the ground truth positions

	File file;
	CHECK_FATAL(file.open(gt_pts_filename, "r") == true);

	GTFile *gt_loader = NULL;
	switch(gt_format)
	{
	case 1:
		gt_loader = new eyecenterGTFile();
		break;
	case 3:
		gt_loader = new bancaGTFile();
		break;
	case 7:
		gt_loader = new cootesGTFile();
		break;
	default:
	   	warning("GT format not implemented.");
	   	return 1;
		break;
	}
	gt_loader->setBOption("verbose", verbose);

	gt_loader->load(&file);

	file.close();

	print("Loaded [%d] ground truth points ...\n", gt_loader->getNPoints());

	int n_ldm_points = gt_loader->getNPoints();
	sPoint2D *ldm_points = gt_loader->getPoints();

	///////////////////////////////////////////////////////////////////
	// Parse the configuration file and set the parameters to ipGeomNorm

	GeomNormParams params;
	CHECK_FATAL(loadGeomNormCfg(cfg_norm_filename, params) == true);
	CHECK_FATAL(params.set(gnormalizer) == true);

	///////////////////////////////////////////////////////////////////
	// Geometric normalize the image and save the result

	CHECK_FATAL(gnormalizer.setGTPoints(ldm_points, n_ldm_points) == true);
	CHECK_FATAL(gnormalizer.process(image) == true);

	const ShortTensor& norm_timage = (const ShortTensor&)gnormalizer.getOutput(0);
	saveImageGTPts(image, ldm_points, n_ldm_points, "original.jpg");
	saveImageGTPts(norm_timage, gnormalizer.getNMPoints(), n_ldm_points, "final.jpg");
	saveImageGTPts(norm_timage, 0, 0, norm_image_filename);

	delete gt_loader;

	print("\nOK\n");

	return 0;
}

