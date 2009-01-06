// Scanning objects
#include "Scanner.h"

#include "PyramidExplorer.h"
#include "MSExplorer.h"
#include "GreedyExplorer.h"

#include "ExhaustiveScaleExplorer.h"
#include "RandomScaleExplorer.h"
#include "SpiralScaleExplorer.h"

#include "ipSWEvaluator.h"
#include "ipSWVariancePruner.h"

#include "OverlapSelector.h"

// Utilities, image processing
#include "CmdLine.h"
#include "FileListCmdOption.h"
#include "Color.h"
#include "general.h"
#include "Image.h"
#include "xtprobeImageFile.h"

// <assert> definition
#include <cassert>

#define CHECK(Test) 				\
{						\
	const bool ret = (Test);		\
	assert(ret == true);			\
}

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Scanning parameters
//////////////////////////////////////////////////////////////////////////

struct Params
{
        // General scanning
        int explorer_type;		// 0 - Pyramid, 1 - Multiscale, 2 - Greedy
	int scale_explorer_type;        // 0 - Exhaustive, 1 - Spiral, 2 - Random, 3 - Mixed
        int min_patt_w, max_patt_w;     // Min/max pattern width/height
        int min_patt_h, max_patt_h;
        float dx, dy, ds;               // Scanning precision factors
        bool stop_at_first_detection;   // Flag
        bool start_with_large_scales;   // Flag

        // Greedy specific
        int greedy_nbest;
        int greedy_perdx;
        int greedy_perdy;
        int greedy_perds;
        int greedy_nsteps;

        // Random specific
        int random_nsamples;

        // Prunning
        bool prune_use_mean;            // Prune sub-windows using the mean
	bool prune_use_stdev;           // Prune sub-windows using the stdev
	double prune_min_mean;          // Prune using mean: min value
	double prune_max_mean;          // Prune using mean: max value
	double prune_min_stdev;         // Prune using stdev: min value
	double prune_max_stdev;         // Prune using stdev: max value

	// Detection merging
	int select_type;		// 0 - Overlap
	int select_merge_type;		// Merge type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence
	bool select_overlap_iterative;	// Overlap: Iterative/One step

        // Debug & log
        bool verbose;			// General verbose flag
	bool save_evaluator_jpg;	// Save candidate sub-windows accepted by the evaluator to jpg
	bool save_pyramid_jpg;		// Save the pyramid images to jpg
};

//////////////////////////////////////////////////////////////////////////
// Set the general options for some explorer
//////////////////////////////////////////////////////////////////////////

void setGeneralOptions(Explorer* explorer, const Params& params)
{
	CHECK(explorer->setIOption("min_patt_w", params.min_patt_w) == true);
	CHECK(explorer->setIOption("max_patt_w", params.max_patt_w) == true);
	CHECK(explorer->setIOption("min_patt_h", params.min_patt_h) == true);
	CHECK(explorer->setIOption("max_patt_h", params.max_patt_h) == true);
	CHECK(explorer->setFOption("ds", params.ds) == true);
	CHECK(explorer->setBOption("StopAtFirstDetection", params.stop_at_first_detection) == true);
	CHECK(explorer->setBOption("StartWithLargeScales", params.start_with_large_scales) == true);
	CHECK(explorer->setBOption("verbose", params.verbose) == true);
}

//////////////////////////////////////////////////////////////////////////
// Save some image to the given file
//////////////////////////////////////////////////////////////////////////

void saveImage(	const Image& image, xtprobeImageFile& xtprobe,
		const char* basename,
		const char* filename)
{
	char str[1024];
	sprintf(str, "%s_%s", basename, filename);

        CHECK(xtprobe.open(str, "w+") == true);
        CHECK(image.saveImage(xtprobe));
        xtprobe.close();
}

//////////////////////////////////////////////////////////////////////////
// Draw some patterns to the original image and save them
//////////////////////////////////////////////////////////////////////////

void savePatterns(	Image& save_image,
			const Image& image, xtprobeImageFile& xtprobe,
			const PatternList& patterns, const char* patterns_name,
			const char* basename, const char* filename)
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
        }
        saveImage(save_image, xtprobe, basename, filename);
}

//////////////////////////////////////////////////////////////////////////
// Save the scanning results (detections, confidence & usage maps) to jpg
//////////////////////////////////////////////////////////////////////////

void saveResults(       const PatternList& detections,
			const PatternSpace& patt_space,
			const Image& image, int image_w, int image_h,
			xtprobeImageFile& xtprobe,
			const char* basename)
{
	Image save_image(image_w, image_h, 3);

        // Print and draw the candidate patterns and the detections
        savePatterns(	save_image, image, xtprobe,
			patt_space.getPatternList(), "candidates",
			basename, "candidates.jpg");

        savePatterns(	save_image, image, xtprobe,
			detections, "detections",
			basename, "detections.jpg");

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

        // Output the usage map
	for (int i = 0; i < image_w; i ++)
                for (int j = 0; j < image_h; j ++)
                {
                        const short value = FixI(scale_confidence_map * confidence_map[i][j]);
                        save_image.set(j, i, 0, value);
			save_image.set(j, i, 1, value);
			save_image.set(j, i, 2, value);
                }
	saveImage(save_image, xtprobe, basename, "usage_map.jpg");

	// Output the confidence map
        for (int i = 0; i < image_w; i ++)
                for (int j = 0; j < image_h; j ++)
                {
                	const short value = usage_map[i][j] == 0x01 ? 255 : 0;
                        save_image.set(j, i, 0, value);
			save_image.set(j, i, 1, value);
			save_image.set(j, i, 2, value);
                }
	saveImage(save_image, xtprobe, basename, "confidence_map.jpg");

        // Output the gradients of the confidence map
        Image grad_ox_conf_image(image_w, image_h, 1);
        Image grad_oy_conf_image(image_w, image_h, 1);
        Image magn_conf_image(image_w, image_h, 1);

	for (int i = 1; i < image_w - 1; i ++)
	{
                for (int j = 1; j < image_h - 1; j ++)
                {
                	int cell[9];

                	cell[0] = save_image.get(j - 1, i - 1, 0);
                	cell[1] = save_image.get(j - 1, i, 0);
                	cell[2] = save_image.get(j - 1, i + 1, 0);
                	cell[3] = save_image.get(j, i - 1, 0);
                	cell[4] = save_image.get(j, i, 0);
                	cell[5] = save_image.get(j, i + 1, 0);
                	cell[6] = save_image.get(j + 1, i - 1, 0);
                	cell[7] = save_image.get(j + 1, i, 0);
                	cell[8] = save_image.get(j + 1, i + 1, 0);

                	const int grad_x = (cell[2] - cell[0] + 2 * (cell[5] - cell[3]) + cell[8] - cell[6]) / 4;
                	const int grad_y = (cell[6] - cell[0] + 2 * (cell[7] - cell[1]) + cell[8] - cell[2]) / 4;

                	grad_ox_conf_image.set(j, i, 0, (255 + grad_x) / 2);
                	grad_oy_conf_image.set(j, i, 0,	(255 + grad_y) / 2);
                	magn_conf_image.set(j, i, 0, getInRange(4 * (abs(grad_x) + abs(grad_y)), 0, 255));
                }
	}

        saveImage(grad_ox_conf_image, xtprobe, basename, "grad_ox_confidence_map.jpg");
        saveImage(grad_oy_conf_image, xtprobe, basename, "grad_oy_confidence_map.jpg");
        saveImage(magn_conf_image, xtprobe, basename, "magn_confidence_map.jpg");
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
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* filename_image;		// Image to scan
        char* filename_model;		// Classifier model to use for sub-window evaluation

        Params params;                  // Scanning parameters

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);
	cmd.info("Object localization.");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("image to scan", &filename_image, "image to scan");
	cmd.addSCmdArg("classifier model", &filename_model, "classifier model");

	cmd.addText("\nScanning options:");
	cmd.addICmdOption("-explorer_type", &params.explorer_type, 0, "explorer type: 0 - Pyramid, 1 - Multiscale, 2 - Greedy");
	cmd.addICmdOption("-scale_explorer_type", &params.scale_explorer_type, 0, "scale explorer type: 0 - Exhaustive, 1 - Spiral, 2 - Random, 3 - Mixed");
	cmd.addICmdOption("-min_patt_w", &params.min_patt_w, 19, "minimum pattern width");
	cmd.addICmdOption("-max_patt_w", &params.max_patt_w, 190, "maximum pattern width");
	cmd.addICmdOption("-min_patt_h", &params.min_patt_h, 19, "minimum pattern height");
	cmd.addICmdOption("-max_patt_h", &params.max_patt_h, 190, "maximum pattern height");
	cmd.addFCmdOption("-dx", &params.dx, 0.2f, "Sub-window Oy position variation");
	cmd.addFCmdOption("-dy", &params.dy, 0.2f, "Sub-window Ox position variation");
	cmd.addFCmdOption("-ds", &params.ds, 1.25f, "Sub-window scale variation");
	cmd.addBCmdOption("-stop_at_first_detection", &params.stop_at_first_detection, false, "stop at first detection");
	cmd.addBCmdOption("-start_with_large_scale", &params.start_with_large_scales, false, "start with large scales");
	cmd.addICmdOption("-greedy_nbest", &params.greedy_nbest, 128, "greedy explorer: number of best detections to refine");
	cmd.addICmdOption("-greedy_perdx", &params.greedy_perdx, 10, "greedy explorer: percentage of Ox candidate's position to vary");
	cmd.addICmdOption("-greedy_perdy", &params.greedy_perdy, 10, "greedy explorer: percentage of Oy candidate's position to vary");
	cmd.addICmdOption("-greedy_perds", &params.greedy_perds, 10, "greedy explorer: percentage of candidate's scale to vary");
	cmd.addICmdOption("-greedy_nsteps", &params.greedy_nsteps, 10, "greedy explorer: number of steps");
	cmd.addICmdOption("-random_nsamples", &params.random_nsamples, 1024, "random scale explorer: number of samples");

	cmd.addText("\nPruning options:");
	cmd.addBCmdOption("-prune_use_mean", &params.prune_use_mean, false, "prune using the mean");
	cmd.addBCmdOption("-prune_use_stdev", &params.prune_use_stdev, false, "prune using the stdev");
	cmd.addDCmdOption("-prune_min_mean", &params.prune_min_mean, 25.0, "prune using the mean: min value");
	cmd.addDCmdOption("-prune_max_mean", &params.prune_max_mean, 225.0, "prune using the mean: max value");
	cmd.addDCmdOption("-prune_min_stdev", &params.prune_min_stdev, 10.0, "prune using the stdev: min value");
	cmd.addDCmdOption("-prune_max_stdev", &params.prune_max_stdev, 125.0, "prune using the stdev: max value");

	cmd.addText("\nCandidate selection options:");
	cmd.addICmdOption("-select_type", &params.select_type, 0, "selector type: 0 - Overlap");
	cmd.addICmdOption("-select_merge_type", &params.select_merge_type, 0, "selector's merging type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence");
	cmd.addBCmdOption("-select_overlap_iterative", &params.select_overlap_iterative, false, "Overlap: Iterative/One step");

        cmd.addText("\nGeneral options:");
	cmd.addBCmdOption("-verbose", &params.verbose, false, "verbose");
	cmd.addBCmdOption("-save_evaluator_jpg", &params.save_evaluator_jpg, false, "save sub-window candidates to jpg ");
	cmd.addBCmdOption("-save_pyramid_jpg", &params.save_pyramid_jpg, false, "save pyramid levels to jpg");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	// Print the arguments
	if (params.verbose == true)
	{
		print("Arguments read: \n");
		print("-----------------------------------------------------------------------------\n");
		print(">>> image: [%s]\n", filename_image);
		print(">>> model: [%s]\n", filename_model);
		print("-----------------------------------------------------------------------------\n");
		print(">>> explorer: [%d]\n", params.explorer_type);
		print(">>> scale explorer: [%d]\n", params.scale_explorer_type);
		print(">>> pattern: [%dx%d] -> [%dx%d]\n",
                        params.min_patt_w, params.min_patt_h, params.max_patt_w, params.max_patt_h);
                print(">>> dx = %f, dy = %f, ds = %f\n", params.dx, params.dy, params.ds);
                print(">>> stop at first detection = %s\n", params.stop_at_first_detection ? "true" : "false");
                print(">>> start with large scales = %s\n", params.start_with_large_scales ? "true" : "false");
                print(">>> greedy: nbest = %d, dx = %d, dy = %d, ds = %d, nsteps = %d\n",
			params.greedy_nbest, params.greedy_perdx, params.greedy_perdy, params.greedy_perds,
			params.greedy_nsteps);
		print(">>> random: nsamples = %d\n", params.random_nsamples);
                print("-----------------------------------------------------------------------------\n");
                print(">>> prune: mean = %s, stdev = %s\n",
                        params.prune_use_mean ? "true" : "false",
                        params.prune_use_stdev ? "true" : "false");
                print(">>> prune params: mean = [%f -> %f], stdev = [%f -> %f]\n",
                        params.prune_min_mean, params.prune_max_mean,
                        params.prune_min_stdev, params.prune_max_stdev);
		print("-----------------------------------------------------------------------------\n");
                print(">>> select: type = %d, merge = %d, overlap iterative = %s\n",
                        params.select_type,
                        params.select_merge_type,
                        params.select_overlap_iterative ? "true" : "false");
		print("-----------------------------------------------------------------------------\n");
		print(">>> verbose: [%s]\n", params.verbose == true ? "true" : "false");
		print(">>> save: evaluator2jpg: [%s]\n", params.save_evaluator_jpg == true ? "true" : "false");
		print(">>> save: pyramid2jpg: [%s]\n", params.save_pyramid_jpg == true ? "true" : "false");
		print("-----------------------------------------------------------------------------\n");
		print("\n");
	}

	params.explorer_type = getInRange(params.explorer_type, 0, 2);
	params.scale_explorer_type = getInRange(params.scale_explorer_type, 0, 2);

	params.select_type = getInRange(params.select_type, 0, 0);
	params.select_merge_type = getInRange(params.select_merge_type, 0, 2);

	///////////////////////////////////////////////////////////////////
        // Load some image
        ///////////////////////////////////////////////////////////////////

        Image image(1, 1, 1);

        xtprobeImageFile xtprobe;
        CHECK(xtprobe.open(filename_image, "r") == true);
        CHECK(image.loadImage(xtprobe) == true);
        xtprobe.close();

	const int image_w = image.getWidth();
	const int image_h = image.getHeight();
	print("Loaded [%s] of [%d x %d].\n", filename_image, image_w, image_h);

	///////////////////////////////////////////////////////////////////
	// Create the scanning objects:
	//	[pruners] + evaluator +
	//	explorer + scale explorers +
	//	selector +
	//	>>> scanner
	///////////////////////////////////////////////////////////////////

        // Evaluator - checks if some sub-window contains a pattern
        ipSWEvaluator evaluator;
        CHECK(evaluator.setClassifier(filename_model) == true);
        CHECK(evaluator.setBOption("verbose", params.verbose) == true);
	CHECK(evaluator.setBOption("saveBuffTensorToJpg", params.save_evaluator_jpg) == true);

        // Pruners - rejects some sub-windows before actually checking is they contain a pattern
	ipSWVariancePruner pruner;
	CHECK(pruner.setBOption("UseMean", params.prune_use_mean) == true);
	CHECK(pruner.setBOption("UseStdev", params.prune_use_stdev) == true);
	pruner.setMinMean(params.prune_min_mean);
	pruner.setMaxMean(params.prune_max_mean);
	pruner.setMinStdev(params.prune_min_stdev);
	pruner.setMaxStdev(params.prune_max_stdev);

        // Explorer - strategy for scanning the image (multiscale, pyramid, greedy ...)
	Explorer* explorer = 0;
	switch (params.explorer_type)
	{
	case 0:	// Pyramid
		explorer = new PyramidExplorer;
		CHECK(explorer->setBOption("savePyramidsToJpg", params.save_pyramid_jpg) == true);
		break;

	case 1:	// Multiscale
                explorer = new MSExplorer;
		break;

	case 2: // Greedy
	default:
		explorer = new GreedyExplorer;
		CHECK(explorer->setIOption("Nbest", params.greedy_nbest) == true);
                CHECK(explorer->setIOption("SWdx", params.greedy_perdx) == true);
                CHECK(explorer->setIOption("SWdy", params.greedy_perdy) == true);
                CHECK(explorer->setIOption("SWds", params.greedy_perds) == true);
                CHECK(explorer->setIOption("NoSteps", params.greedy_nsteps) == true);
		break;
	}
	setGeneralOptions(explorer, params);

	// ScaleExplorers - fixed scale scanning methods
	ExhaustiveScaleExplorer scale_explorer_ex;
	CHECK(scale_explorer_ex.setFOption("dx", params.dx) == true);
	CHECK(scale_explorer_ex.setFOption("dy", params.dy) == true);
	CHECK(scale_explorer_ex.setBOption("verbose", params.verbose) == true);

	SpiralScaleExplorer scale_explorer_sp;
	CHECK(scale_explorer_sp.setFOption("dx", params.dx) == true);
	CHECK(scale_explorer_sp.setFOption("dy", params.dy) == true);
	CHECK(scale_explorer_sp.setBOption("verbose", params.verbose) == true);

	RandomScaleExplorer scale_explorer_rd;
	CHECK(scale_explorer_rd.setIOption("NSamples", params.random_nsamples) == true);
	CHECK(scale_explorer_rd.setBOption("verbose", params.verbose) == true);

	const int n_scale_explorers = 3;
	ScaleExplorer* scale_explorers[n_scale_explorers] =
		{
			&scale_explorer_ex, &scale_explorer_sp, &scale_explorer_rd
		};
	const char* str_scale_explorers[n_scale_explorers] =
		{
			"ExhaustiveScaleExplorer", "SpiralScaleExplorer", "RandomScaleExplorer"
		};

	// Merging techniques for the OverlapSelector
	AveragePatternMerger pattern_merge_avg;
	ConfWeightedPatternMerger pattern_merge_confWeighted;
	MaxConfPatternMerger pattern_merge_maxConf;

	const int n_pattern_mergers = 3;
	PatternMerger* pattern_mergers[n_pattern_mergers] =
		{
			&pattern_merge_avg, &pattern_merge_confWeighted, &pattern_merge_maxConf
		};
	const char* str_pattern_mergers[n_pattern_mergers] =
		{
			"Average", "Confidence Weighted", "Maximum Confidence"
		};

	// Selectors - select the best pattern sub-windows from the candidates
	OverlapSelector selector;
	selector.setMerger(pattern_mergers[params.select_merge_type]);
	CHECK(selector.setBOption("verbose", params.verbose) == true);
	CHECK(selector.setBOption("iterative", params.select_overlap_iterative) == true);

	// Scanner - main scanning object, contains the ROIs
	Scanner scanner;
	CHECK(scanner.setBOption("verbose", params.verbose) == true);
        scanner.deleteAllROIs();
	CHECK(scanner.getNoROIs() == 0);
	//CHECK(scanner.addROI(0, 0, 400, 400) == true);
	//CHECK(scanner.getNoROIs() == 1);

	///////////////////////////////////////////////////////////////////
	// Assembly the scanning object and process the image
	///////////////////////////////////////////////////////////////////

	print("-----------------------------------------------------------------------------\n");
	print("Running the explorer ...\n");
	print("-----------------------------------------------------------------------------\n");

	// Assembly the main scanning object as desired
	explorer->deleteAllSWPruners();
	CHECK(explorer->setSWEvaluator(&evaluator) == true);
	CHECK(explorer->addSWPruner(&pruner) == true);

	CHECK(scanner.setExplorer(explorer) == true);
	CHECK(scanner.setSelector(&selector) == true);

	// Initialize processing
	CHECK(scanner.init(image) == true);
	const int n_scales = explorer->getNoScales();

        // Set for each scale the feature extractors (<ipCore>s)
        //      [0/NULL] means the original image will be used as features!
        switch (params.explorer_type)
	{
	case 0:	// Pyramid
                for (int j = 0; j < n_scales; j ++)
                {
                        CHECK(explorer->setScalePruneIp(j, 0) == true);
                        CHECK(explorer->setScaleEvaluationIp(j, 0) == true);
                }
		break;

	case 1:	// Multiscale
	case 2: // Greedy
	default:
		CHECK(explorer->setScalePruneIp(0) == true);
                CHECK(explorer->setScaleEvaluationIp(0) == true);
		break;
	}

        // Set for each scale the scanning type (exhaustive, random, spiral ...)
        for (int j = 0; j < n_scales; j ++)
        {
        	int index = 0;
        	switch (params.scale_explorer_type)
        	{
		case 0:	// Exhaustive
			index = 0;
			break;

		case 1:	// Spiral
			index = 1;
			break;

		case 2:	// Random
			index = 2;
			break;

		case 3:	// Mixed
		default:
			index = rand() % n_scale_explorers;
			break;
        	}
                CHECK(explorer->setScaleExplorer(j, scale_explorers[index]) == true);

		const sSize& scale = explorer->getScale(j);
		print(">>> scale [%dx%d] -> %s\n", scale.w, scale.h, str_scale_explorers[index]);
        }

        // Scan the image and get the results
        CHECK(scanner.process(image) == true);

        print("No of sub-windows: pruned = %d, scanned = %d, accepted = %d\n",
                scanner.getNoPrunnedSWs(),
                scanner.getNoScannedSWs(),
                scanner.getNoAcceptedSWs());

        ///////////////////////////////////////////////////////////////////
        // Output the final results
        ///////////////////////////////////////////////////////////////////

        print("-----------------------------------------------------------------------------\n");
        print("Results ... \n");
        print("-----------------------------------------------------------------------------\n");

	char basename[1024];
	getBaseFileName(filename_image, basename);

	char save_basename[2048];
	sprintf(save_basename,
		"%s_%s",
		basename,
		(params.explorer_type == 0) ?
			"pyramid" : ((params.explorer_type == 1) ? "multiscale" : "greedy"));

        saveResults(	selector.getPatterns(),		// final patterns
			explorer->getPatternSpace(),	// pattern space from where the final patterns where selected
			image, image_w, image_h,
			xtprobe,
			save_basename);

	// Cleanup
	delete explorer;

        // OK
	return 0;
}

