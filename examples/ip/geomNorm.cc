#include "torch5spro.h"

using namespace Torch;

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
	CHECK_FATAL(gnormalizer.setBOption("verbose", verbose) == true);

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

	const int n_ldm_points = gt_loader->getNPoints();
	const sPoint2D *ldm_points = gt_loader->getPoints();

	///////////////////////////////////////////////////////////////////
	// Parse the configuration file and set the parameters to ipGeomNorm

	CHECK_FATAL(gnormalizer.loadCfg(cfg_norm_filename) == true);
	CHECK_FATAL(gnormalizer.setGTFile(gt_loader) == true);

	///////////////////////////////////////////////////////////////////
	// Geometric normalize the image and save the result

	CHECK_FATAL(gnormalizer.process(image) == true);

	const ShortTensor& norm_timage = (const ShortTensor&)gnormalizer.getOutput(0);
	saveImageGTPts(image, ldm_points, n_ldm_points, "original.jpg");
	saveImageGTPts(norm_timage, gnormalizer.getNMPoints(), n_ldm_points, "final.jpg");
	saveImageGTPts(norm_timage, 0, 0, norm_image_filename);

	delete gt_loader;

	print("\nOK\n");

	return 0;
}

