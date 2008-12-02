#include "Scanner.h"

#include "PyramidExplorer.h"

#include "ExhaustiveScaleExplorer.h"
#include "RandomScaleExplorer.h"
#include "SpiralScaleExplorer.h"

#include "ipSWEvaluator.h"
#include "ipSWDummyPruner.h"

#include "DummySelector.h"

#include "Image.h"
#include "xtprobeImageFile.h"
#include "Color.h"
#include "general.h"

#include <cassert>

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Save some image
//////////////////////////////////////////////////////////////////////////

void saveImage(const Image& image, xtprobeImageFile& xtprobe, const char* filename)
{
        assert(xtprobe.open(filename, "w+") == true);
        assert(image.saveImage(xtprobe));
        xtprobe.close();
}

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main()
{
        const char* imagefilename = "../data/images/003_1_1.pgm";
        const char* modelfilename = "../data/models/facedetection/frontal/mct4.cascade";

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
	print("Loaded [%s] of [%d x %d].\n",
                imagefilename, image_w, image_h);

	///////////////////////////////////////////////////////////////////
	// Create the scanning objects:
	//	[pruners] + evaluator +
	//	explorer + scale explorers +
	//	selector +
	//	>>> scanner
	///////////////////////////////////////////////////////////////////

        ipSWEvaluator evaluator;
        assert(evaluator.setClassifier(modelfilename) == true);

	ipSWDummyPruner pruner1(ipSWDummyPruner::RejectNone);
	ipSWDummyPruner pruner2(ipSWDummyPruner::RejectNone);
	ipSWDummyPruner pruner3(ipSWDummyPruner::RejectNone);

	PyramidExplorer explorer;
	explorer.deleteAllSWPruners();
	assert(explorer.setSWEvaluator(&evaluator) == true);
	assert(explorer.addSWPruner(&pruner1) == true);
	assert(explorer.addSWPruner(&pruner2) == true);
	assert(explorer.addSWPruner(&pruner3) == true);

	// Different types of fixed scale scanning
	ExhaustiveScaleExplorer scale_explorer_ex;
	SpiralScaleExplorer scale_explorer_sp;
	RandomScaleExplorer scale_explorer_rd;

	const int n_scale_explorers = 3;
	ScaleExplorer* scale_explorers[n_scale_explorers] =
		{
			&scale_explorer_ex, &scale_explorer_sp, &scale_explorer_rd
		};
	const char* str_scale_explorers[n_scale_explorers] =
		{
			"ExhaustiveScaleExplorer", "SpiralScaleExplorer", "RandomScaleExplorer"
		};

	// This object will select the best pattern sub-windows from the candidates
	DummySelector selector;

	// This is the main scanning object
	Scanner scanner;

	///////////////////////////////////////////////////////////////////
	// Set some scanning parameters
	///////////////////////////////////////////////////////////////////

	// <Evaluator>
	assert(evaluator.setBOption("verbose", true) == true);
	assert(evaluator.setBOption("saveBuffTensorToJpg", true) == true);

        // <Scanner>
	assert(scanner.setBOption("verbose", true) == true);

        // <Scanner>'s ROIs
	scanner.deleteAllROIs();
	assert(scanner.getNoROIs() == 0);

	//sRect2D roi;
	//roi.x = 0; roi.y = 0; roi.w = 400; roi.h = 400;
	//assert(scanner.addROI(roi) == true);
	//assert(scanner.getNoROIs() == 1);

	//roi.x = 200; roi.y = 200; roi.w = 400; roi.h = 400;
	//assert(scanner.addROI(roi) == true);
	//assert(scanner.getNoROIs() == 2);

        // <PyramidExplorer>
	assert(explorer.setIOption("min_patt_w", 19) == true);
	assert(explorer.setIOption("max_patt_w", 190) == true);
	assert(explorer.setIOption("min_patt_h", 19) == true);
	assert(explorer.setIOption("max_patt_h", 190) == true);
	assert(explorer.setFOption("ds", 0.5f) == true);
	assert(explorer.setBOption("StopAtFirstDetection", false) == true);
	assert(explorer.setBOption("StartWithLargeScales", true) == true);
	assert(explorer.setBOption("verbose", true) == true);
	assert(explorer.setBOption("savePyramidsToJpg", true) == true);

	// <ExhaustiveScaleExplore>
	assert(scale_explorer_ex.setFOption("dx", 0.2f) == true);
	assert(scale_explorer_ex.setFOption("dy", 0.2f) == true);
	assert(scale_explorer_ex.setBOption("verbose", true) == true);

	// <SpiralScaleExplorer>
	assert(scale_explorer_sp.setFOption("dx", 0.2f) == true);
	assert(scale_explorer_sp.setFOption("dy", 0.2f) == true);
	assert(scale_explorer_sp.setBOption("verbose", true) == true);

	// <RandomScaleExplorer>
	assert(scale_explorer_rd.setIOption("NSamples", 256) == true);
	assert(scale_explorer_rd.setBOption("verbose", true) == true);

	///////////////////////////////////////////////////////////////////
	// Assembly the scanning object and process the image
	///////////////////////////////////////////////////////////////////

	print("--------------------------------------------------\n");
	print("Testing the pyramid explorer ...\n");
	print("--------------------------------------------------\n");

	// Assembly the main scanning object as desired
	assert(scanner.setExplorer(&explorer) == true);
	assert(scanner.setSelector(&selector) == true);

	// Initialize processing
	assert(scanner.init(image) == true);
	const int n_scales = explorer.getNoScales();

        // Set for each scale the feature extractors (<ipCore>s)
        //      [0/NULL] means the original image will be used as features!
        assert(explorer.setScalePruneIp(0) == false);
	assert(explorer.setScaleEvaluationIp(0) == false);
        for (int j = 0; j < n_scales; j ++)
        {
                assert(explorer.setScalePruneIp(j, 0) == true);
                assert(explorer.setScaleEvaluationIp(j, 0) == true);
        }

        // Set for each scale the scanning type (exhaustive, random, spiral)
        for (int j = 0; j < n_scales; j ++)
        {
                const int index = 0;// Exhaustive scale search only (rand() % n_scale_explorers)
                assert(explorer.setScaleExplorer(j, scale_explorers[index]) == true);

                const sSize& scale = explorer.getScale(j);
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

        const PatternList& detections = selector.getPatterns();
        const int no_detections = detections.size();

        print("--------------------------------------------------\n");
        print("Results ... \n");
        print("--------------------------------------------------\n");

        // Print the detections
        print("No of detections: %d:\n", detections.size());
        for (int i = 0; i < no_detections; i ++)
        {
                const Pattern& detection = detections.get(i);

                print("---> [%d, %d] : [%dx%d] with [%f] confidence\n",
                        detection.m_x, detection.m_y, detection.m_w, detection.m_h,
                        detection.m_confidence);
        }

        // Draw the detections over the image and save it
        Image detection_image(image_w, image_h, 3);
        detection_image.copyFrom(image);
        for (int i = 0; i < no_detections; i ++)
        {
                const Pattern& detection = detections.get(i);
                detection_image.drawRect(
                        detection.m_x, detection.m_y, detection.m_w, detection.m_h,
                        red);
        }
        saveImage(detection_image, xtprobe, "detections.jpg");

        // Get the confidence and the usage maps
        const PatternSpace& patt_space = explorer.getPatternSpace();
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

        // OK
	return 0;
}

