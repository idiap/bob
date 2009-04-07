#include "xtprobeImageFile.h"
#include "Image.h"
#include "FileListCmdOption.h"
#include "CmdLine.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	// Set source images
	FileListCmdOption* image_files = new FileListCmdOption("images", "source images");
	image_files->isArgument(true);

	// Directory where to load images from
	char*	dir_images;

	// Image extension
	char*	ext_images;

	// Number of subwindows to generate
	int	n_wnds;

	// Directory to save .wnd files
	char*	dir_wnds;

	// Desired output size (the subwindows will respect thess minimum values and ratio)
	int	out_width, out_height;

	// Options
	bool	verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);
	cmd.info("Generating subwindows to be used with ImageScanDataSet.");

	cmd.addText("\nArguments:");
	cmd.addCmdOption(image_files);
	cmd.addSCmdArg("image directory", &dir_images, "directory where to load images from");
	cmd.addSCmdArg("image extension", &ext_images, "image extension");
	cmd.addICmdArg("no. of subwindows", &n_wnds, "number of subwindows to generate");
	cmd.addSCmdArg(".wnd directory", &dir_wnds, "directory where to save the .wnd files");
	cmd.addICmdArg("width", &out_width, "desired width");
	cmd.addICmdArg("height", &out_height, "desired height");

	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	// Check arguments
	if (image_files->n_files < 1)
	{
		print("Error: no image file provided!\n");
		return 1;
	}

	xtprobeImageFile xtprobe;
	Image image(1, 1, 1);

	srand((unsigned int)time(0));
	const int n_wnds_per_image = n_wnds / image_files->n_files;

	// Generate subwindows for each image ...
	for (int i = 0; i < image_files->n_files; i ++)
	{
		char str[1024];

		// Load the image
		sprintf(str, "%s/%s.%s", dir_images, image_files->file_names[i], ext_images);
		CHECK_FATAL(xtprobe.load(image, str) == true);
		const int image_w = image.getWidth();
		const int image_h = image.getHeight();
		if (image_w < out_width || image_h < out_height)
		{
			print("Image [%s] is too small! Ignoring...\n", image_files->file_names[i]);
			continue;
		}
		print("Loaded [%s] of %dx%d size.\n", image_files->file_names[i], image_w, image_h);

		sprintf(str, "%s/%s.wnd", dir_wnds, image_files->file_names[i]);
		File fwnd;
		CHECK_FATAL(fwnd.open(str, "w") == true);
		CHECK_FATAL(fwnd.write(&n_wnds_per_image, sizeof(int), 1) == 1);

		// Generate randomly the subwindows
		for (int j = 0; j < n_wnds_per_image; j ++)
		{
			const short x = rand() % (image_w - out_width);
			const short y = rand() % (image_h - out_height);
			const double scale = 1.0 + 0.001 * (rand() %
						min(	1000 * (image_w - x - out_width) / out_width,
							1000 * (image_h - y - out_height) / out_height));
			const short w = min(FixI(scale * out_width), image_w - x);
			const short h = min(FixI(scale * out_height), image_h - y);

			if (verbose == true)
			{
				print("\t[%d/%d]: (%d, %d) - %dx%d\n", j + 1, n_wnds_per_image, x, y, w, h);
			}
			CHECK_FATAL(fwnd.write(&x, sizeof(short), 1) == 1);
			CHECK_FATAL(fwnd.write(&y, sizeof(short), 1) == 1);
			CHECK_FATAL(fwnd.write(&w, sizeof(short), 1) == 1);
			CHECK_FATAL(fwnd.write(&h, sizeof(short), 1) == 1);
		}

		fwnd.close();
	}

	print("\nOK\n");

   	return 0;
}

