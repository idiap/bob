#include "torch5spro.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Save some image to the given file
//////////////////////////////////////////////////////////////////////////

void saveImage(	const Image& image, xtprobeImageFile& xtprobe,
		const char* basename,
		const char* filename)
{
	char str[1024];
	sprintf(str, "%s_%s", basename, filename);
        CHECK_FATAL(xtprobe.save(image, str) == true);
}

//////////////////////////////////////////////////////////////////////////
// Draw some patterns to the original image and save them
//////////////////////////////////////////////////////////////////////////

void savePatterns(	Image& save_image, const Image& image, xtprobeImageFile& xtprobe,
			const PatternList& patterns,
			const char* patterns_name, const char* basename, const char* filename)
{
	const int no_patterns = patterns.size();

	print("No of %s: [[[[ %d ]]]]\n", patterns_name, no_patterns);
        save_image.copyFrom(image);
        for (int i = 0; i < no_patterns; i ++)
        {
                const Pattern& pattern = patterns.get(i);
                //print("---> [%d, %d] : [%dx%d] with [%f] confidence\n",
                //      pattern.m_x, pattern.m_y, pattern.m_w, pattern.m_h,
                //      pattern.m_confidence);
                save_image.drawRect(pattern.m_x, pattern.m_y, pattern.m_w, pattern.m_h, red);
                save_image.drawRect(pattern.m_x - 1, pattern.m_y - 1, pattern.m_w + 2, pattern.m_h + 2, red);
                save_image.drawRect(pattern.m_x + 1, pattern.m_y + 1, pattern.m_w - 2, pattern.m_h - 2, red);
        }
        saveImage(save_image, xtprobe, basename, filename);
}

//////////////////////////////////////////////////////////////////////////
// Save the detections results (before and after merging) to jpg
//////////////////////////////////////////////////////////////////////////

void saveResults(       const PatternList& detections, const PatternList& candidates,
			const Image& image, xtprobeImageFile& xtprobe, const char* basename)
{
	Image save_image(image.getWidth(), image.getHeight(), 3);

        // Print and draw the candidate patterns and the detections
        savePatterns(	save_image, image, xtprobe,
			candidates, "candidates",
			basename, "candidates.jpg");

        savePatterns(	save_image, image, xtprobe,
			detections, "detections",
			basename, "detections.jpg");
}

/////////////////////////////////////////////////////////////
// Extract basename from filename
/////////////////////////////////////////////////////////////

void getBaseFileName(const char* filename, char* basename)
{
        strcpy(basename, filename);
        char* extension = (char*) strrchr(basename, '.');
        if (extension != 0)
                *extension = '\0';

        char* separator = (char*) rindex(basename, '/');
        if (separator != 0)
        {
                separator++;
                strcpy(basename, separator);
        }
}

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	// Set options
        char* filename_image;		// Image to scan
        char* filename_params;		// Scanning parameters

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);
	cmd.info("Face detection.");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("image", &filename_image, "image to scan");
	cmd.addSCmdArg("params", &filename_params, "face detector parameters file");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	// Load the image
	Image image(1, 1, 1);
        xtprobeImageFile xtprobe;
        CHECK_FATAL(xtprobe.load(image, filename_image) == true);
	print("Loaded [%s] of [%d x %d].\n", filename_image, image.getWidth(), image.getHeight());

	// Load the face finder
	FaceFinder ffinder;
	CHECK_FATAL(ffinder.reset(filename_params) == true);

	// Process the image
	print("-----------------------------------------------------------------------------\n");
	print("Running the face detection ...\n");
	print("-----------------------------------------------------------------------------\n");

	MTimer timer;
        timer.reset();

        // Scan the image and get the results
        CHECK_FATAL(ffinder.process(image) == true);

        const Scanner& scanner = ffinder.getScanner();
	const Explorer& explorer = scanner.getExplorer();
	const int n_scales = explorer.getNoScales();
        for (int j = 0; j < n_scales; j ++)
        {
        	const sSize& scale = explorer.getScale(j);
		print(">>> scale [%dx%d]\n", scale.w, scale.h);
        }

        print("Scanning time: %ld usecs\n", timer.stop());
        print("No of sub-windows: pruned = %d, scanned = %d, accepted = %d\n",
                scanner.getNoPrunnedSWs(),
                scanner.getNoScannedSWs(),
                scanner.getNoAcceptedSWs());

        // Output the final results
        print("-----------------------------------------------------------------------------\n");
        print("Results ... \n");
        print("-----------------------------------------------------------------------------\n");

	char basename[1024];
	getBaseFileName(filename_image, basename);

	FaceFinder::Params params;
	CHECK_FATAL(params.load(filename_params) == true);

	char save_basename[2048];
	sprintf(save_basename,
		"%s_%s",
		basename,
		(params.explorer_type == 0) ?
			"pyramid" : ((params.explorer_type == 1) ? "multiscale" : "context"));

        saveResults(	scanner.getPatterns(),		// final (merged) detections
			explorer.getPatterns(),		// candidate (not merged) detections
			image, xtprobe, save_basename);

	// OK
	return 0;
}

