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
#include "MeanShiftSelector.h"

// Utilities, image processing
#include "CmdLine.h"
#include "FileListCmdOption.h"
#include "Color.h"
#include "general.h"
#include "Image.h"
#include "xtprobeImageFile.h"

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
	int select_type;		// 0 - Overlap, 1 - MeanShift
	int select_merge_type;		// Merge type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence
	bool select_overlap_iterative;	// Overlap: Iterative/One step
	int select_min_surf_overlap;	// Overlap: Minimum surface overlap to merge

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
	CHECK_FATAL(explorer->setIOption("min_patt_w", params.min_patt_w) == true);
	CHECK_FATAL(explorer->setIOption("max_patt_w", params.max_patt_w) == true);
	CHECK_FATAL(explorer->setIOption("min_patt_h", params.min_patt_h) == true);
	CHECK_FATAL(explorer->setIOption("max_patt_h", params.max_patt_h) == true);
	CHECK_FATAL(explorer->setFOption("ds", params.ds) == true);
	CHECK_FATAL(explorer->setBOption("StopAtFirstDetection", params.stop_at_first_detection) == true);
	CHECK_FATAL(explorer->setBOption("StartWithLargeScales", params.start_with_large_scales) == true);
	CHECK_FATAL(explorer->setBOption("verbose", params.verbose) == true);
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

        CHECK_FATAL(xtprobe.save(image, str) == true);
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
			const PatternList& candidates,
			const Image& image, int image_w, int image_h,
			xtprobeImageFile& xtprobe,
			const char* basename)
{
	Image save_image(image_w, image_h, 3);

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
	cmd.addICmdOption("-select_type", &params.select_type, 1, "selector type: 0 - Overlap, 1 - MeanShift");
	cmd.addICmdOption("-select_merge_type", &params.select_merge_type, 0, "selector's merging type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence");
	cmd.addBCmdOption("-select_overlap_iterative", &params.select_overlap_iterative, false, "Overlap: Iterative/One step");
	cmd.addICmdOption("-select_min_surf_overlap", &params.select_min_surf_overlap, 60, "Overlap: minimum surface overlap to merge");

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
                print(">>> greedy: dx = %d, dy = %d, ds = %d, nsteps = %d\n",
			params.greedy_perdx, params.greedy_perdy, params.greedy_perds,
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
                print(">>> select: type = %d, merge = %d, overlap iterative = %s, minSurfOverlap = %d\n",
                        params.select_type,
                        params.select_merge_type,
                        params.select_overlap_iterative ? "true" : "false",
                        params.select_min_surf_overlap);
		print("-----------------------------------------------------------------------------\n");
		print(">>> verbose: [%s]\n", params.verbose == true ? "true" : "false");
		print(">>> save: evaluator2jpg: [%s]\n", params.save_evaluator_jpg == true ? "true" : "false");
		print(">>> save: pyramid2jpg: [%s]\n", params.save_pyramid_jpg == true ? "true" : "false");
		print("-----------------------------------------------------------------------------\n");
		print("\n");
	}

	params.explorer_type = getInRange(params.explorer_type, 0, 2);
	params.scale_explorer_type = getInRange(params.scale_explorer_type, 0, 2);

	params.select_type = getInRange(params.select_type, 0, 1);
	params.select_merge_type = getInRange(params.select_merge_type, 0, 2);

	///////////////////////////////////////////////////////////////////
        // Load some image
        ///////////////////////////////////////////////////////////////////

        Image image(1, 1, 1);

        xtprobeImageFile xtprobe;
        CHECK_FATAL(xtprobe.load(image, filename_image) == true);

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
        CHECK_FATAL(evaluator.setClassifier(filename_model) == true);
        CHECK_FATAL(evaluator.setBOption("verbose", params.verbose) == true);
	CHECK_FATAL(evaluator.setBOption("saveBuffTensorToJpg", params.save_evaluator_jpg) == true);

        // Pruners - rejects some sub-windows before actually checking is they contain a pattern
	ipSWVariancePruner pruner;
	CHECK_FATAL(pruner.setBOption("UseMean", params.prune_use_mean) == true);
	CHECK_FATAL(pruner.setBOption("UseStdev", params.prune_use_stdev) == true);
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
		CHECK_FATAL(explorer->setBOption("savePyramidsToJpg", params.save_pyramid_jpg) == true);
		break;

	case 1:	// Multiscale
                explorer = new MSExplorer;
		break;

	case 2: // Greedy
	default:
		explorer = new GreedyExplorer;
		CHECK_FATAL(explorer->setIOption("SWdx", params.greedy_perdx) == true);
                CHECK_FATAL(explorer->setIOption("SWdy", params.greedy_perdy) == true);
                CHECK_FATAL(explorer->setIOption("SWds", params.greedy_perds) == true);
                CHECK_FATAL(explorer->setIOption("NoSteps", params.greedy_nsteps) == true);
		break;
	}
	setGeneralOptions(explorer, params);

	// ScaleExplorers - fixed scale scanning methods
	ExhaustiveScaleExplorer scale_explorer_ex;
	CHECK_FATAL(scale_explorer_ex.setFOption("dx", params.dx) == true);
	CHECK_FATAL(scale_explorer_ex.setFOption("dy", params.dy) == true);
	CHECK_FATAL(scale_explorer_ex.setBOption("verbose", params.verbose) == true);

	SpiralScaleExplorer scale_explorer_sp;
	CHECK_FATAL(scale_explorer_sp.setFOption("dx", params.dx) == true);
	CHECK_FATAL(scale_explorer_sp.setFOption("dy", params.dy) == true);
	CHECK_FATAL(scale_explorer_sp.setBOption("verbose", params.verbose) == true);

	RandomScaleExplorer scale_explorer_rd;
	CHECK_FATAL(scale_explorer_rd.setIOption("NSamples", params.random_nsamples) == true);
	CHECK_FATAL(scale_explorer_rd.setBOption("verbose", params.verbose) == true);

	const int n_scale_explorers = 3;
	ScaleExplorer* scale_explorers[n_scale_explorers] =
		{
			&scale_explorer_ex,
			&scale_explorer_sp,
			&scale_explorer_rd
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
			&pattern_merge_avg,
			&pattern_merge_confWeighted,
			&pattern_merge_maxConf
		};

	// Selectors - select the best pattern sub-windows from the candidates
	OverlapSelector selector_ov;
	selector_ov.setMerger(pattern_mergers[params.select_merge_type]);
	CHECK_FATAL(selector_ov.setIOption("minSurfOverlap", params.select_min_surf_overlap) == true);
	CHECK_FATAL(selector_ov.setBOption("iterative", params.select_overlap_iterative) == true);
	CHECK_FATAL(selector_ov.setBOption("verbose", params.verbose) == true);
	CHECK_FATAL(selector_ov.setBOption("onlySurfOverlaps", true) == true);
	CHECK_FATAL(selector_ov.setBOption("onlyMaxSurf", false) == true);
	CHECK_FATAL(selector_ov.setBOption("onlyMaxConf", false) == true);

	MeanShiftSelector selector_ms;
	CHECK_FATAL(selector_ms.setBOption("verbose", params.verbose) == true);

	const int n_selectors = 2;
	Selector* selectors[n_selectors] =
		{
			&selector_ov,
			&selector_ms
		};

	// Scanner - main scanning object, contains the ROIs
	Scanner scanner;
	CHECK_FATAL(scanner.setBOption("verbose", params.verbose) == true);
        scanner.deleteAllROIs();
	CHECK_FATAL(scanner.getNoROIs() == 0);
	//CHECK_FATAL(scanner.addROI(0, 0, 400, 400) == true);
	//CHECK_FATAL(scanner.getNoROIs() == 1);

	///////////////////////////////////////////////////////////////////
	// Assembly the scanning object and process the image
	///////////////////////////////////////////////////////////////////

	print("-----------------------------------------------------------------------------\n");
	print("Running the explorer ...\n");
	print("-----------------------------------------------------------------------------\n");

	// Assembly the main scanning object as desired
	explorer->deleteAllSWPruners();
	CHECK_FATAL(explorer->setSWEvaluator(&evaluator) == true);
	CHECK_FATAL(explorer->addSWPruner(&pruner) == true);

	CHECK_FATAL(scanner.setExplorer(explorer) == true);
	CHECK_FATAL(scanner.setSelector(selectors[params.select_type]) == true);

	// Initialize processing
	CHECK_FATAL(scanner.init(image) == true);
	const int n_scales = explorer->getNoScales();

        // Set for each scale the feature extractors (<ipCore>s)
        //      [0/NULL] means the original image will be used as features!
        switch (params.explorer_type)
	{
	case 0:	// Pyramid
                for (int j = 0; j < n_scales; j ++)
                {
                        CHECK_FATAL(explorer->setScalePruneIp(j, 0) == true);
                        CHECK_FATAL(explorer->setScaleEvaluationIp(j, 0) == true);
                }
		break;

	case 1:	// Multiscale
	case 2: // Greedy
	default:
		CHECK_FATAL(explorer->setScalePruneIp(0) == true);
                CHECK_FATAL(explorer->setScaleEvaluationIp(0) == true);
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
                CHECK_FATAL(explorer->setScaleExplorer(j, scale_explorers[index]) == true);

		const sSize& scale = explorer->getScale(j);
		print(">>> scale [%dx%d] -> %s\n", scale.w, scale.h, str_scale_explorers[index]);
        }

        // Scan the image and get the results
        CHECK_FATAL(scanner.preprocess(image) == true);
        CHECK_FATAL(scanner.process(image) == true);

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

        saveResults(	selectors[params.select_type]->getPatterns(),		// final patterns
			explorer->getPatterns(),	// pattern space from where the final patterns where selected
			image, image_w, image_h,
			xtprobe,
			save_basename);

	// Cleanup
	delete explorer;

        // OK
	return 0;
}

