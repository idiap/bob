
// Scanning objects
#include "Scanner.h"

#include "PyramidExplorer.h"
#include "MSExplorer.h"
#include "GreedyExplorer.h"

#include "ExhaustiveScaleExplorer.h"
#include "RandomScaleExplorer.h"
#include "SpiralScaleExplorer.h"

#include "ipSWEvaluator.h"
#include "ipSWDummyPruner.h"

#include "DummySelector.h"

// Utilities, image processing
#include "CmdLine.h"
#include "FileListCmdOption.h"
#include "Color.h"
#include "general.h"
#include "Image.h"
#include "xtprobeImageFile.h"

// <assert> definition
#include <cassert>

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Set the general options for some explorer
//////////////////////////////////////////////////////////////////////////

void setGeneralOptions(Explorer* explorer, bool verbose)
{
	assert(explorer->setIOption("min_patt_w", 19) == true);
	assert(explorer->setIOption("max_patt_w", 190) == true);
	assert(explorer->setIOption("min_patt_h", 19) == true);
	assert(explorer->setIOption("max_patt_h", 190) == true);
	assert(explorer->setFOption("ds", 0.5f) == true);
	assert(explorer->setBOption("StopAtFirstDetection", false) == true);
	assert(explorer->setBOption("StartWithLargeScales", true) == true);
	assert(explorer->setBOption("verbose", verbose) == true);
}

//////////////////////////////////////////////////////////////////////////
// Build and initialize the explorer algorithm to use
//////////////////////////////////////////////////////////////////////////

Explorer* BuildPyramidExplorer(bool verbose, bool save_pyramid_jpg)
{
	PyramidExplorer* explorer = new PyramidExplorer();

	setGeneralOptions(explorer, verbose);
	assert(explorer->setBOption("savePyramidsToJpg", save_pyramid_jpg) == true);

	return explorer;
}

Explorer* BuildMultiscaleExplorer(bool verbose)
{
	MSExplorer* explorer = new MSExplorer();

	setGeneralOptions(explorer, verbose);

	return explorer;
}

Explorer* BuildGreedyExplorer(bool verbose)
{
	GreedyExplorer* explorer = new GreedyExplorer();

	setGeneralOptions(explorer, verbose);
	assert(explorer->setIOption("Nbest", 128) == true);

	return explorer;
}

//////////////////////////////////////////////////////////////////////////
// Save some image to the given file
//////////////////////////////////////////////////////////////////////////

void saveImage(const Image& image, xtprobeImageFile& xtprobe, const char* filename)
{
        assert(xtprobe.open(filename, "w+") == true);
        assert(image.saveImage(xtprobe));
        xtprobe.close();
}

//////////////////////////////////////////////////////////////////////////
// Save the scanning results (detections, confidence & usage maps) to jpg
//////////////////////////////////////////////////////////////////////////

void saveResults(       const PatternList& detections,
			const PatternSpace& patt_space,
			const Image& image, int image_w, int image_h,
			xtprobeImageFile& xtprobe)
{
        const int no_detections = detections.size();

        Image detection_image(image_w, image_h, 3);
        detection_image.copyFrom(image);

        // Print and draw the detections
        print("No of detections: [[[[ %d ]]]]\n", no_detections);
        for (int i = 0; i < no_detections; i ++)
        {
                const Pattern& detection = detections.get(i);

                //print("---> [%d, %d] : [%dx%d] with [%f] confidence\n",
                //      detection.m_x, detection.m_y, detection.m_w, detection.m_h,
                //      detection.m_confidence);

                detection_image.drawRect(
			detection.m_x, detection.m_y, detection.m_w, detection.m_h, red);
        }
        saveImage(detection_image, xtprobe, "detections.jpg");

        // Get the confidence and the usage maps
        int** confidence_map = patt_space.getConfidenceMap();
        unsigned char** usage_map = patt_space.getUsageMap();

        int max_confidence_map = 0;
        for (int i = 0; i < image_w; i ++)
                for (int j = 0; j < image_h; j ++)
                {
                        max_confidence_map = max(max_confidence_map, confidence_map[i][j]);
                }
        const double scale_confidence_map = 255.0 / (max_confidence_map + 1.0);

        // Output the confidence and the usage maps
        Image confidence_image(image_w, image_h, 1);
        Image usage_image(image_w, image_h, 1);

         for (int i = 0; i < image_w; i ++)
                for (int j = 0; j < image_h; j ++)
                {
                        usage_image.set(j, i, 0, usage_map[i][j] == 0x01 ? 255 : 0);
                        confidence_image.set(j, i, 0,
                                (short)(scale_confidence_map * confidence_map[i][j] + 0.5));
                }

        saveImage(confidence_image, xtprobe, "confidence_map.jpg");
        saveImage(usage_image, xtprobe, "usage_map.jpg");
}

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
        const char* imagefilename = "../data/images/003_1_1.pgm";
        const char* modelfilename = "../data/models/facedetection/frontal/mct4.cascade";

        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* filename_image;		// Image to scan
        char* filename_model;		// Classifier model to use for sub-window evaluation

	bool verbose;			// General verbose flag
	int explorer_type;		// 0 - Pyramid, 1 - Multiscale, 2 - Greedy
	bool save_evaluator_jpg;	// Save candidate sub-windows accepted by the evaluator to jpg
	bool save_pyramid_jpg;		// Save the pyramid images to jpg

	//FileListCmdOption* file_list = new FileListCmdOption("test", "test list");// Just for testing
	//file_list->isArgument(true);

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);
	cmd.info("Object localization.");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("image to scan", &filename_image, "image to scan");
	cmd.addSCmdArg("classifier model", &filename_model, "classifier model");
	//cmd.addCmdOption(file_list);

	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addICmdOption("-explorer_type", &explorer_type, 0, "explorer type");
	cmd.addBCmdOption("-save_evaluator_jpg", &save_evaluator_jpg, false, "save sub-window candidates to jpg ");
	cmd.addBCmdOption("-save_pyramid_jpg", &save_pyramid_jpg, false, "save pyramid levels to jpg");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	// Print the arguments
	if (verbose == true)
	{
		print("Arguments read: \n");

		print(">>> image: [%s]\n", filename_image);
		print(">>> model: [%s]\n", filename_model);
		print(">>> verbose: [%s]\n", verbose == true ? "true" : "false");
		print(">>> explorer: [%d]\n", explorer_type);
		print(">>> save: evaluator2jpg: [%s]\n", save_evaluator_jpg == true ? "true" : "false");
		print(">>> save: pyramid2jpg: [%s]\n", save_pyramid_jpg == true ? "true" : "false");

		//print(">>> file_list: size [%d]\n", file_list->n_files);
		//for (int i = 0; i < file_list->n_files; i ++)
		//{
		//	print(">>> file_list: file [%d/%d] [%s]\n",
		//		i + 1, file_list->n_files, file_list->file_names[i]);
		//}

		print("\n");
	}

	explorer_type = getInRange(explorer_type, 0, 2);

	///////////////////////////////////////////////////////////////////
        // Load some image
        ///////////////////////////////////////////////////////////////////

        Image image(1, 1, 1);

        xtprobeImageFile xtprobe;
        assert(xtprobe.open(imagefilename, "r") == true);
        assert(image.loadImage(xtprobe) == true);
        xtprobe.close();

	const int image_w = image.getWidth();
	const int image_h = image.getHeight();
	print("Loaded [%s] of [%d x %d].\n", imagefilename, image_w, image_h);

	///////////////////////////////////////////////////////////////////
	// Create the scanning objects:
	//	[pruners] + evaluator +
	//	explorer + scale explorers +
	//	selector +
	//	>>> scanner
	///////////////////////////////////////////////////////////////////

        // Evaluator - checks if some sub-window contains a pattern
        ipSWEvaluator evaluator;
        assert(evaluator.setClassifier(modelfilename) == true);
        assert(evaluator.setBOption("verbose", verbose) == true);
	assert(evaluator.setBOption("saveBuffTensorToJpg", save_evaluator_jpg) == true);

        // Pruners - rejects some sub-windows before actually checking is they contain a pattern
	ipSWDummyPruner pruner1(ipSWDummyPruner::RejectNone);
	ipSWDummyPruner pruner2(ipSWDummyPruner::RejectNone);
	ipSWDummyPruner pruner3(ipSWDummyPruner::RejectNone);

        // Explorer - strategy for scanning the image (multiscale, pyramid, greedy ...)
	Explorer* explorer = 0;
	switch (explorer_type)
	{
	case 0:	// Pyramid
		explorer = BuildPyramidExplorer(verbose, save_pyramid_jpg);
		break;

	case 1:	// Multiscale
		explorer = BuildMultiscaleExplorer(verbose);
		break;

	case 2: // Greedy
	default:
		explorer = BuildGreedyExplorer(verbose);
		break;
	}

	explorer->deleteAllSWPruners();
	assert(explorer->setSWEvaluator(&evaluator) == true);
	assert(explorer->addSWPruner(&pruner1) == true);
	assert(explorer->addSWPruner(&pruner2) == true);
	assert(explorer->addSWPruner(&pruner3) == true);

	// Fixed scale scanning methods
	ExhaustiveScaleExplorer scale_explorer_ex;
	assert(scale_explorer_ex.setFOption("dx", 0.2f) == true);
	assert(scale_explorer_ex.setFOption("dy", 0.2f) == true);
	assert(scale_explorer_ex.setBOption("verbose", verbose) == true);

	SpiralScaleExplorer scale_explorer_sp;
	assert(scale_explorer_sp.setFOption("dx", 0.2f) == true);
	assert(scale_explorer_sp.setFOption("dy", 0.2f) == true);
	assert(scale_explorer_sp.setBOption("verbose", verbose) == true);

	RandomScaleExplorer scale_explorer_rd;
	assert(scale_explorer_rd.setIOption("NSamples", 256) == true);
	assert(scale_explorer_rd.setBOption("verbose", verbose) == true);

	const int n_scale_explorers = 3;
	ScaleExplorer* scale_explorers[n_scale_explorers] =
		{
			&scale_explorer_ex, &scale_explorer_sp, &scale_explorer_rd
		};
	const char* str_scale_explorers[n_scale_explorers] =
		{
			"ExhaustiveScaleExplorer", "SpiralScaleExplorer", "RandomScaleExplorer"
		};

	// Selector - select the best pattern sub-windows from the candidates
	DummySelector selector;

	// Scanner - main scanning object
	Scanner scanner;
	assert(scanner.setBOption("verbose", verbose) == true);

        // and it contains the ROIs
        scanner.deleteAllROIs();
	assert(scanner.getNoROIs() == 0);
	//assert(scanner.addROI(0, 0, 400, 400) == true);
	//assert(scanner.getNoROIs() == 1);
	//assert(scanner.addROI(200, 200, 400, 400) == true);
	//assert(scanner.getNoROIs() == 2);

	///////////////////////////////////////////////////////////////////
	// Assembly the scanning object and process the image
	///////////////////////////////////////////////////////////////////

	print("--------------------------------------------------\n");
	print("Running the explorer ...\n");
	print("--------------------------------------------------\n");

	// Assembly the main scanning object as desired
	assert(scanner.setExplorer(explorer) == true);
	assert(scanner.setSelector(&selector) == true);

	// Initialize processing
	assert(scanner.init(image) == true);
	const int n_scales = explorer->getNoScales();

        // Set for each scale the feature extractors (<ipCore>s)
        //      [0/NULL] means the original image will be used as features!
        assert(explorer->setScalePruneIp(0) == (explorer_type != 0));			// Multiscale specific
	assert(explorer->setScaleEvaluationIp(0) == (explorer_type != 0));		// Multiscale specific
        for (int j = 0; j < n_scales; j ++)
        {
                assert(explorer->setScalePruneIp(j, 0) == (explorer_type == 0));	// Pyramid specific
                assert(explorer->setScaleEvaluationIp(j, 0) == (explorer_type == 0));	// Pyramid specific
        }

        // Set for each scale the scanning type (exhaustive, random, spiral ...)
        for (int j = 0; j < n_scales; j ++)
        {
                const int index = 0;// Exhaustive scale search only (rand() % n_scale_explorers)
                assert(explorer->setScaleExplorer(j, scale_explorers[index]) == true);

		const sSize& scale = explorer->getScale(j);
		print(">>> scale [%dx%d] -> %s\n", scale.w, scale.h, str_scale_explorers[index]);
        }

        // Scan the image and get the results
        assert(scanner.process(image) == true);

        print("No of sub-windows: accepted = %d, pruned = %d, scanned = %d\n",
                scanner.getNoAcceptedSWs(),
                scanner.getNoPrunnedSWs(),
                scanner.getNoScannedSWs());

        ///////////////////////////////////////////////////////////////////
        // Output the final results
        ///////////////////////////////////////////////////////////////////

        print("--------------------------------------------------\n");
        print("Results ... \n");
        print("--------------------------------------------------\n");

        saveResults(	selector.getPatterns(),		// final patterns
			explorer->getPatternSpace(),	// pattern space from where the final patterns where selected
			image, image_w, image_h,
			xtprobe);

	// Cleanup
	delete explorer;

        // OK
	return 0;
}

