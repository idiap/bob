#include "ImageScanDataSet.h"
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

	// Set source .wnd
	FileListCmdOption* wnd_files = new FileListCmdOption("wnd", "source subwindows");
	wnd_files->isArgument(true);

	// Directory where to load images from
	char*	dir_images;

	// Image extension
	char*	ext_images;

	// Directory to load .wnd files from
	char*	dir_wnds;

	// Desired output size
	int	out_width, out_height;

	// Options
	bool	save;		// Save subwindows to jpg
	bool	verbose;	// Verbose

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);
	cmd.info("Testing ImageScanDataSet.");

	cmd.addText("\nArguments:");
	cmd.addCmdOption(image_files);
	cmd.addCmdOption(wnd_files);
	cmd.addSCmdArg("image directory", &dir_images, "directory where to load images from");
	cmd.addSCmdArg("image extension", &ext_images, "image extension");
	cmd.addSCmdArg(".wnd directory", &dir_wnds, "directory where to load the .wnd files from");
	cmd.addICmdArg("width", &out_width, "desired width");
	cmd.addICmdArg("height", &out_height, "desired height");

	cmd.addBCmdOption("-save", &save, false, "save subwindows to .jpg");
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
	if (image_files->n_files != wnd_files->n_files)
	{
		print("Error: mismatch between the number of image files and wnd files!\n");
		return 1;
	}

	// Create the main ImageScanDataSet object
	ImageScanDataSet iscandataset(	image_files->n_files,
					dir_images, image_files->file_names, ext_images,
					dir_wnds, wnd_files->file_names,
					out_width, out_height, 1);

	const long n_examples = iscandataset.getNoExamples();
	print("No of examples: %ld\n", n_examples);

	// Test the targets (rejection of samples)
	DoubleTensor reject_target(1), accept_target(1);
	reject_target.fill(-1.0);
	accept_target.fill(1.0);

	for (long i = 0; i < n_examples; i ++)
	{
		if (verbose == true)
		{
			print("Rejecting [%d/%d]\r", i + 1, n_examples);
		}
		CHECK_FATAL(((DoubleTensor*)iscandataset.getTarget(i))->get(0) > 0.0);
		iscandataset.setTarget(i, &reject_target);
	}

	for (long i = 0; i < n_examples; i ++)
	{
		if (verbose == true)
		{
			print("Accepting [%d/%d]\r", i + 1, n_examples);
		}
		CHECK_FATAL(((DoubleTensor*)iscandataset.getTarget(i))->get(0) < 0.0);
		iscandataset.setTarget(i, &accept_target);
		CHECK_FATAL(((DoubleTensor*)iscandataset.getTarget(i))->get(0) > 0.0);
	}

	print("Targets tested: OK                               \n");

	// Test the subwindow sampling
	for (long i = 0; i < n_examples; i ++)
	{
		if (verbose == true)
		{
			print("Testing examples [%d/%d]\r", i + 1, n_examples);
		}
		Image* image = (Image*)iscandataset.getExample(i);

		if (save == true)
		{
			char str[512];
			sprintf(str, "%d_%d.jpg", i + 1, n_examples);

			xtprobeImageFile xtprobe;
			CHECK_FATAL(xtprobe.save(*image, str) == true);
		}
	}

	print("Examples tested: OK                              \n");

   	return 0;
}

