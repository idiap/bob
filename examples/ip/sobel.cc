#include "ipSobel.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "CmdLine.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	char* image_filename = 0;

	CmdLine cmd;
	cmd.info("Testing program for histograms.\n");
	//cmd.addICmdArg("test_type", &test_type, "0 - image, 1 - fill(0)");
	cmd.addSCmdArg("-image", &image_filename, "input image");

	cmd.read(argc, argv);

	xtprobeImageFile xtprobe;
	Image image(1, 1, 3);
	Image Gimage;
	//Image Gy_image;
	//Image Mag_image;

	// Load the image to play with
	//const char* imagefilename = "../data/images/Jaded2.pgm";
	CHECK_FATAL(xtprobe.open(image_filename, "r") == true);
	CHECK_FATAL(image.loadImage(xtprobe) == true);
	xtprobe.close();
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image.getWidth(), image.getHeight(), image.getNPlanes());

	// create sobel class
	ipSobel sobel;// = new ipSobel();

	CHECK_FATAL(sobel.process(image) == true);
	CHECK_FATAL(sobel.getNOutputs() == 3);

	// Save it to some file
	Gimage.resize(image.getWidth(), image.getHeight(), image.getNPlanes());
	CHECK_FATAL(Gimage.copyFrom(sobel.getOutput(0)) == true);

	char str[200];
	sprintf(str, "Image_Gx.jpg");
	CHECK_FATAL(xtprobe.open(str, "w") == true);
	CHECK_FATAL(Gimage.saveImage(xtprobe) == true);
	xtprobe.close();

	CHECK_FATAL(Gimage.copyFrom(sobel.getOutput(1)) == true);
	sprintf(str, "Image_Gy.jpg");
	CHECK_FATAL(xtprobe.open(str, "w") == true);
	CHECK_FATAL(Gimage.saveImage(xtprobe) == true);
	xtprobe.close();

	CHECK_FATAL(Gimage.copyFrom(sobel.getOutput(2)) == true);
	sprintf(str, "Image_Gmag.jpg");
	CHECK_FATAL(xtprobe.open(str, "w") == true);
	CHECK_FATAL(Gimage.saveImage(xtprobe) == true);
	xtprobe.close();

	print("\nOK\n");
	//delete sobel;
	//delete image;
	return 0;
}

