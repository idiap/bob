#include "Scanner.h"

#include "MSExplorer.h"
#include "PyramidExplorer.h"
#include "GreedyExplorer.h"

#include "ExhaustiveScaleExplorer.h"
#include "RandomScaleExplorer.h"
#include "SpiralScaleExplorer.h"

#include "ipSWDummyEvaluator.h"
#include "ipSWDummyPruner.h"
#include "ipIntegralImage.h"

#include "DummySelector.h"

#include "Pattern.h"

#include "Image.h"
#include "general.h"

#include <cassert>
#include <cstdlib>

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main()
{
	const int image_w = 800;
	const int image_h = 600;

	///////////////////////////////////////////////////////////////////
	// Generate a random image
	///////////////////////////////////////////////////////////////////

	srand((unsigned int)time(0));
	Image image(image_w, image_h, 1);
	for (int j = 0; j < image_h; j ++)
		for (int i = 0; i < image_w; i ++)
		{
			image.set(j, i, 0, rand() % 256);
		}

	///////////////////////////////////////////////////////////////////
	// Create the scanning objects:
	//	scale feature processing +
	//	[pruners] + evaluator +
	//	explorer + scale explorers +
	//	selector +
	//	scanner
	///////////////////////////////////////////////////////////////////

	ipIntegralImage iimage;

	ipSWDummyEvaluator evaluator(19, 19, ipSWDummyEvaluator::PassRandom);

	ipSWDummyPruner pruner1(ipSWDummyPruner::RejectRandom);
	ipSWDummyPruner pruner2(ipSWDummyPruner::RejectRandom);
	ipSWDummyPruner pruner3(ipSWDummyPruner::RejectRandom);

	// Different type of scale-space scanning (exploration) methods
	MSExplorer explorer_ms;
	PyramidExplorer explorer_pr;
	GreedyExplorer explorer_gd;

	// Make the explorers use the prunning and evaluation methods above
	const int n_explorers = 3;
	Explorer* explorers[n_explorers] =
		{
			&explorer_ms, &explorer_pr, &explorer_gd
		};
	const char* str_explorers[n_explorers] =
		{
			"MultiScale", "Pyramid", "Greedy"
		};

	for (int i = 0; i < n_explorers; i ++)
	{
		assert(explorers[i]->setSWEvaluator(&evaluator) == true);
		assert(explorers[i]->addSWPruner(&pruner1) == true);
		assert(explorers[i]->addSWPruner(&pruner2) == true);
		assert(explorers[i]->addSWPruner(&pruner3) == true);
	}

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

	// Set some ROIs to use
	scanner.deleteAllROIs();
	assert(scanner.getNoROIs() == 0);

	sRect2D roi;
	roi.x = 0; roi.y = 0; roi.w = 400; roi.h = 400;
	assert(scanner.addROI(roi) == true);
	assert(scanner.getNoROIs() == 1);

	roi.x = 200; roi.y = 200; roi.w = 400; roi.h = 400;
	assert(scanner.addROI(roi) == true);
	assert(scanner.getNoROIs() == 2);

	///////////////////////////////////////////////////////////////////
	// Set some scanning parameters
	///////////////////////////////////////////////////////////////////

	assert(scanner.setBOption("verbose", true) == true);

	// The common <Explorer> parameters
	for (int i = 0; i < n_explorers; i ++)
	{
		Explorer* explorer = explorers[i];

		assert(explorer->setIOption("min_patt_w", 57) == true);
		assert(explorer->setIOption("max_patt_w", 85) == true);
		assert(explorer->setIOption("min_patt_h", 57) == true);
		assert(explorer->setIOption("max_patt_h", 190) == true);

		assert(explorer->setFOption("ds", 0.1f) == true);
		assert(explorer->setBOption("StopAtFirstDetection", true) == true);
		assert(explorer->setBOption("StartWithLargeScales", true) == true);

		assert(explorer->setBOption("verbose", true) == true);
	}

	// <MSExplorer> specific parameters
	// - none

	// <PyramidExplorer> specific parameters
	// - none

	// <GreedyExplorer> specific parameters
	assert(explorer_gd.setIOption("Nbest", 128) == true);

	// <ExhaustiveScaleExplore> specific parameters
	assert(scale_explorer_ex.setFOption("dx", 0.2f) == true);
	assert(scale_explorer_ex.setFOption("dy", 0.2f) == true);
	assert(scale_explorer_ex.setBOption("verbose", true) == true);

	// <SpiralScaleExplorer> specific parameters
	assert(scale_explorer_sp.setFOption("dx", 0.2f) == true);
	assert(scale_explorer_sp.setFOption("dy", 0.2f) == true);
	assert(scale_explorer_sp.setBOption("verbose", true) == true);

	// <RandomScaleExplorer> specific parameters
	assert(scale_explorer_rd.setIOption("NSamples", 256) == true);
	assert(scale_explorer_rd.setBOption("verbose", true) == true);

	///////////////////////////////////////////////////////////////////
	// Assembly the scanning object and process the image
	///////////////////////////////////////////////////////////////////

	for (int i = 0; i < n_explorers; i ++)
	{
		print("--------------------------------------------------\n");
		print("Testing [%s] explorer ...\n", str_explorers[i]);
		print("--------------------------------------------------\n");

		Explorer* explorer = explorers[i];

		// Assembly the main scanning object as desired
		assert(scanner.setExplorer(explorer) == true);
		assert(scanner.setSelector(&selector) == true);
		assert(scanner.init(image) == true);

		// Set for each scale the <ipCore>s for prunning and evaluation
		const bool one_ipCore = i != 1;
		const bool multiple_ipCores = i == 1;

		assert(explorer->setScalePruneIp(&iimage) == one_ipCore);
		assert(explorer->setScaleEvaluationIp(&iimage) == one_ipCore);

		const int n_scales = explorer->getNoScales();
		for (int j = 0; j < n_scales; j ++)
		{
			assert(explorer->setScalePruneIp(j, &iimage) == multiple_ipCores);
			assert(explorer->setScaleEvaluationIp(j, &iimage) == multiple_ipCores);
		}

		// Set for each scale the scanning type (exhaustive, random, spiral)
		for (int j = 0; j < n_scales; j ++)
		{
			const int index = rand() % n_scale_explorers;
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

		//const PatternList& patterns = selector.getPatterns();
		//print("No of stored sub-windows = %d\n", patterns.size());
	}

	return 0;
}

