#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* image_filename = 0;
	int test_type = 0;	// 0 - image, 1 - fill(0)

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for histograms.\n");
	cmd.addICmdArg("test_type", &test_type, "0 - image, 1 - fill(0)");
	cmd.addSCmdOption("-image", &image_filename, "", "input image");

	cmd.read(argc, argv);

	// Load or generate the image
	Image image(1, 1, 1);
	switch (test_type)
	{
	case 0:	// Image
		{
			xtprobeImageFile xtprobe;
			CHECK_FATAL(xtprobe.load(image, image_filename) == true);
		}
		break;

	case 1:	// Homogenous image
	default:
		{
			srand((unsigned int)time(0));
			const int px = rand() % 256;
			print("Generating a homogenous image with pixel value = %d, \n", px);

			CHECK_FATAL(image.resize(32, 32, 1) == true);
			image.fill(px);
		}
		break;
	}

	print("Processing image [width = %d, height = %d, nplanes = %d] ...\n",
		image.size(1), image.size(0), image.size(2));
	print("\nHISTOGRAM:\n");

	// Process the image and get the results
	ipHisto ip_histo;
	CHECK_FATAL(ip_histo.process(image) == true);
	CHECK_FATAL(ip_histo.getNOutputs() == 1);
	CHECK_FATAL(ip_histo.getOutput(0).getDatatype() == Tensor::Int);
	const IntTensor& out_histo = (const IntTensor&) ip_histo.getOutput(0);

	// Print the histogram
	const int n_planes = image.getNPlanes();
	if (n_planes == 1)
	{
		for (int i = 0; i < 256; i ++)
		{
			print("[%d/255]: \t%d\n", i, out_histo.get(i, 0));
		}
	}
	else
	{
		for (int i = 0; i < 256; i ++)
		{
			print("[%d/255]: ", i);
			for (int p = 0; p < n_planes; p ++)
			{
				print("%d, ", out_histo.get(i, p));
			}
			print("\n");
		}
	}

	// Check its validity
	const int n_pixels = image.size(0) * image.size(1) * image.size(2);
	int sum_histo = 0;
	for (int i = 0; i < 256; i ++)
	{
		for (int p = 0; p < n_planes; p ++)
		{
			sum_histo += out_histo.get(i, p);
		}
	}
	CHECK_FATAL(sum_histo == n_pixels);

	print("\nOK\n");

	return 0;
}

