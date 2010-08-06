#include "FaceFinder.h"
#include "Scanner.h"
#include "core/Image.h"
#include "Explorer.h"
#include "PyramidExplorer.h"
#include "MSExplorer.h"
#include "ContextExplorer.h"
#include "TrackContextExplorer.h"
#include "ScaleExplorer.h"
#include "ExhaustiveScaleExplorer.h"
#include "SpiralScaleExplorer.h"
#include "RandomScaleExplorer.h"
#include "OverlapSelector.h"
#include "MeanShiftSelector.h"
#include "DummySelector.h"
#include "core/spCoreChain.h"
#include "CascadeMachine.h"
#include "ipSWEvaluator.h"
#include "ipSWVariancePruner.h"
#include "ip/ipLBPBitmap.h"
#include "ip/ipLBP4R.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

FaceFinder::Params::Params(const char* filename)
	:	f_model(0),
		f_thres(0.0),
		use_f_thres(false),

		explorer_type(1),		// Multiscale
		scale_explorer_type(0),		// Exhaustive

		min_patt_w(0), max_patt_w(4096),
		min_patt_h(0), max_patt_h(4096),
		dx(0.1), dy(0.1), ds(1.1),
		stop_at_first_detection(false),
		start_with_large_scales(false),
		random_nsamples(8192),

		prune_use_mean(false),
		prune_use_stdev(false),
		prune_min_mean(25.0),
		prune_max_mean(225.0),
		prune_min_stdev(10.0),
		prune_max_stdev(125.0),

		prep_ii(true),
		prep_hlbp(false),

		select_type(0),			// Overlap
		select_overlap_type(0),		// Average
		select_overlap_iterative(false),// Not iterative overlap merging
		select_overlap_min_surf(60),	// 60% overlap to merge two detections

		context_model(0),
		context_type(0),		// Full context sampling

		verbose(false)
{
	// Add the associated parameters to be parsed
	cmd_file.addText("\nModel options:");
	cmd_file.addSCmd("model", &f_model, "face classifier model");
	cmd_file.addDCmdOption("-model_thres", &f_thres, 0.0, "threshold of the face classifier model");
	cmd_file.addBCmdOption("-model_use_thres", &use_f_thres, false, "use the given threshold");

	cmd_file.addText("\nScanning options:");
	cmd_file.addICmdOption("-explorer_type", &explorer_type, 0, "explorer type: 0 - Pyramid, 1 - Multiscale, 2 - Context");
	cmd_file.addICmdOption("-scale_explorer_type", &scale_explorer_type, 0, "scale explorer type: 0 - Exhaustive, 1 - Spiral, 2 - Random");
	cmd_file.addICmdOption("-min_patt_w", &min_patt_w, 19, "minimum pattern width");
	cmd_file.addICmdOption("-max_patt_w", &max_patt_w, 190, "maximum pattern width");
	cmd_file.addICmdOption("-min_patt_h", &min_patt_h, 19, "minimum pattern height");
	cmd_file.addICmdOption("-max_patt_h", &max_patt_h, 190, "maximum pattern height");
	cmd_file.addDCmdOption("-dx", &dx, 0.2f, "Sub-window Oy position variation");
	cmd_file.addDCmdOption("-dy", &dy, 0.2f, "Sub-window Ox position variation");
	cmd_file.addDCmdOption("-ds", &ds, 1.25f, "Sub-window scale variation");
	cmd_file.addBCmdOption("-stop_at_first_detection", &stop_at_first_detection, false, "stop at first detection");
	cmd_file.addBCmdOption("-start_with_large_scale", &start_with_large_scales, false, "start with large scales");
	cmd_file.addICmdOption("-random_nsamples", &random_nsamples, 1024, "random scale explorer: number of samples");

	cmd_file.addText("\nPreprocessing options:");
	cmd_file.addBCmdOption("-prep_ii", &prep_ii, false, "Compute integral image");
	cmd_file.addBCmdOption("-prep_hlbp", &prep_hlbp, false, "HLBP: compute LBP4R bitmaps");

	cmd_file.addText("\nPruning options:");
	cmd_file.addBCmdOption("-prune_use_mean", &prune_use_mean, false, "prune using the mean");
	cmd_file.addBCmdOption("-prune_use_stdev", &prune_use_stdev, false, "prune using the stdev");
	cmd_file.addDCmdOption("-prune_min_mean", &prune_min_mean, 25.0, "prune using the mean: min value");
	cmd_file.addDCmdOption("-prune_max_mean", &prune_max_mean, 225.0, "prune using the mean: max value");
	cmd_file.addDCmdOption("-prune_min_stdev", &prune_min_stdev, 10.0, "prune using the stdev: min value");
	cmd_file.addDCmdOption("-prune_max_stdev", &prune_max_stdev, 125.0, "prune using the stdev: max value");

	cmd_file.addText("\nCandidate selection options:");
	cmd_file.addICmdOption("-select_type", &select_type, 1, "selector type: 0 - Overlap, 1 - MeanShift, 2 - No merge");
	cmd_file.addICmdOption("-select_overlap_type", &select_overlap_type, 0, "selector's merging type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence");
	cmd_file.addBCmdOption("-select_overlap_iterative", &select_overlap_iterative, false, "Overlap: Iterative/One step");
	cmd_file.addICmdOption("-select_overlap_min_surf", &select_overlap_min_surf, 60, "Overlap: minimum surface overlap to merge");

	cmd_file.addText("\nContext-based model:");
	cmd_file.addSCmdOption("-context_model", &context_model, "", "Face context-based model");
	cmd_file.addICmdOption("-context_type", &context_type, 1, "Context type (0 - Full, 1 - Axis)");

        cmd_file.addText("\nGeneral options:");
	cmd_file.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Load the parameter values
	load(filename);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

FaceFinder::Params::~Params()
{
}

/////////////////////////////////////////////////////////////////////////
// Load/save the configuration to/from a CmdFile-like file

bool FaceFinder::Params::load(const char* filename)
{
	if (filename == 0)
		return false;
	cmd_file.read(filename, true);
	return true;
}

bool FaceFinder::Params::save(const char* filename) const
{
	return cmd_file.write(filename);
}

/////////////////////////////////////////////////////////////////////////
// Display current parameter values

bool FaceFinder::Params::print() const
{
	File file;
	return file.open(stdout) && cmd_file.write(file);
}

/////////////////////////////////////////////////////////////////////////
// Constructors

FaceFinder::FaceFinder(FaceFinder::Params* params)
	:	m_scanner(0),
		m_sp_prep(0),
		m_explorer(0),
		m_scale_explorer(0)
{
	reset(params);
}

FaceFinder::FaceFinder(const char* filename)
	:	m_scanner(0),
		m_sp_prep(0),
		m_explorer(0),
		m_scale_explorer(0)
{
	reset(filename);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

FaceFinder::~FaceFinder()
{

}

/////////////////////////////////////////////////////////////////////////

namespace Private
{
	// Build a SW evaluator given the parametrization
	ipSWEvaluator* buildSWEvaluator(const FaceFinder::Params* params)
	{
		// Set options
		ipSWEvaluator* evaluator = manage(new ipSWEvaluator);
		if (	evaluator->setClassifier(params->f_model) == false ||
			evaluator->setBOption("verbose", params->verbose) == false)
		{
			warning("FaceFinder::buildSWEvaluator - error!\n");
			return 0;
		}

		// Change the model threshold if required
		if (params->use_f_thres)
		{
		   	Classifier *classifier_ = &evaluator->getClassifier();
			classifier_->setThreshold(params->f_thres);

			/*
			CascadeMachine* cascade = dynamic_cast<CascadeMachine*>(&evaluator->getClassifier());
			if (cascade == 0 || cascade->getNoStages() < 1)
			{
				warning("FaceFinder::reset - received a face model that is not a cascade!\n");
				return 0;
			}
			cascade->setThreshold(cascade->getNoStages() - 1, params->f_thres);
			*/
		}

		// OK
		return evaluator;
	}

	// Build a SW pruner given the parametrization
	ipSWVariancePruner* buildSWPruner(const FaceFinder::Params* params)
	{
		// Set options
		ipSWVariancePruner* pruner = manage(new ipSWVariancePruner);
		if (	pruner->setBOption("UseMean", params->prune_use_mean) == false ||
			pruner->setBOption("UseStdev", params->prune_use_stdev) == false)
		{
			warning("FaceFinder::buildSWPruner - error!\n");
			return 0;
		}
		pruner->setMinMean(params->prune_min_mean);
		pruner->setMaxMean(params->prune_max_mean);
		pruner->setMinStdev(params->prune_min_stdev);
		pruner->setMaxStdev(params->prune_max_stdev);

		// OK
		return pruner;
	}

	// Build an explorer given the parametrization
	Explorer* buildExplorer(const FaceFinder::Params* params)
	{
		// Build the selected explorer
		Explorer* explorer = 0;
		switch (params->explorer_type)
		{
		case 0:	// Pyramid
			explorer = manage(new PyramidExplorer);
			break;

		case 1:	// Multiscale
			explorer = manage(new MSExplorer);
			break;

		case 2: // Context
			explorer = manage(new ContextExplorer);
			if (	((ContextExplorer*)explorer)->setContextModel(params->context_model) == false ||
				((ContextExplorer*)explorer)->setIOption("ctx_type", params->context_type) == false)
			{
				return 0;
			}
			((ContextExplorer*)explorer)->setMode(ContextExplorer::Scanning);
			break;
		
		case 3: // Track context
		default:
			explorer = manage(new TrackContextExplorer);
			if (	((TrackContextExplorer*)explorer)->setContextModel(params->context_model) == false ||
				((TrackContextExplorer*)explorer)->setIOption("ctx_type", params->context_type) == false)
			{
				return 0;
			}
			((TrackContextExplorer*)explorer)->setMode(ContextExplorer::Scanning);
			break;
		}
		if (	explorer->setIOption("min_patt_w", params->min_patt_w) == false ||
			explorer->setIOption("max_patt_w", params->max_patt_w) == false ||
			explorer->setIOption("min_patt_h", params->min_patt_h) == false ||
			explorer->setIOption("max_patt_h", params->max_patt_h) == false ||
			explorer->setFOption("ds", params->ds) == false ||
			explorer->setBOption("StopAtFirstDetection", params->stop_at_first_detection) == false ||
			explorer->setBOption("StartWithLargeScales", params->start_with_large_scales) == false ||
			explorer->setBOption("verbose", params->verbose) == false)
		{
			warning("FaceFinder::buildExplorer - error!\n");
			return 0;
		}

		// Assembly the explorer as desired
		explorer->deleteAllSWPruners();
		if (	params->prune_use_mean && params->prune_use_stdev &&
			explorer->setSWEvaluator(buildSWEvaluator(params)) == false)
		{
			warning("FaceFinder::buildExplorer - error\n");
			return 0;
		}
		if (explorer->setSWEvaluator(buildSWEvaluator(params)) == false)
		{
			warning("FaceFinder::buildExplorer - error!\n");
			return 0;
		}

		// OK
		return explorer;
	}

	// Build a scale explorer given the parametrization
	ScaleExplorer* buildScaleExplorer(const FaceFinder::Params* params)
	{
		switch (params->scale_explorer_type)
		{
		case 0:	// Exhaustive
			{
				ExhaustiveScaleExplorer* scale_explorer = manage(new ExhaustiveScaleExplorer);
				if (	scale_explorer->setFOption("dx", params->dx) == false ||
					scale_explorer->setFOption("dy", params->dy) == false ||
					scale_explorer->setBOption("verbose", params->verbose) == false)
				{
					warning("FaceFinder::buildScaleExplorer - error!\n");
					return 0;
				}

				return scale_explorer;
			}

		case 1:	// Spiral
			{
				SpiralScaleExplorer* scale_explorer = manage(new SpiralScaleExplorer);
				if (	scale_explorer->setFOption("dx", params->dx) == false ||
					scale_explorer->setFOption("dy", params->dy) == false ||
					scale_explorer->setBOption("verbose", params->verbose) == false)
				{
					warning("FaceFinder::buildScaleExplorer - error!\n");
					return 0;
				}

				return scale_explorer;
			}

		case 2:	// Random
		default:
			{
				RandomScaleExplorer* scale_explorer = manage(new RandomScaleExplorer);
				if (	scale_explorer->setIOption("NSamples", params->random_nsamples) == false ||
					scale_explorer->setBOption("verbose", params->verbose) == false)
				{
					warning("FaceFinder::buildScaleExplorer - error!\n");
					return 0;
				}

				return scale_explorer;
			}
        	}
	}

	// Build a selector given the parametrization
	PatternMerger* buildMerger(const FaceFinder::Params* params)
	{
		switch (params->select_overlap_type)
		{
		case 0:
			return manage(new AveragePatternMerger);

		case 1:
			return manage(new ConfWeightedPatternMerger);

		case 2:
		default:
			return manage(new MaxConfPatternMerger);
		}
	}
	Selector* buildSelector(const FaceFinder::Params* params)
	{
		switch (params->select_type)
		{
		case 0:	// Overlap
			{
				OverlapSelector* selector = manage(new OverlapSelector);
				selector->setMerger(buildMerger(params));
				if (	selector->setIOption("minSurfOverlap", params->select_overlap_min_surf) == false ||
					selector->setBOption("iterative", params->select_overlap_iterative) == false ||
					selector->setBOption("verbose", params->verbose) == false ||
					selector->setBOption("onlySurfOverlaps", true) == false ||
					selector->setBOption("onlyMaxSurf", false) == false ||
					selector->setBOption("onlyMaxConf", false) == false)
				{
					warning("FaceFinder::buildSelector - error!\n");
					return 0;
				}

				return selector;
			}
			break;

		case 1: // Adaptive Mean Shift
			{
				MeanShiftSelector* selector = manage(new MeanShiftSelector);
				if (	selector->setBOption("verbose", params->verbose) == false ||
					selector->setIOption("kernel", 1) == false)	// 0 - constant, 1 - liniar, 2 - quadratic
				{
					warning("FaceFinder::buildSelector - error!\n");
					return 0;
				}

				return selector;
			}
			break;

		case 2:	// No merge
		default:
			return manage(new DummySelector);
		}
	}
}

/////////////////////////////////////////////////////////////////////////
// Change the scanning parameters

bool FaceFinder::reset(FaceFinder::Params* params)
{
	if (params == 0)
	{
		return false;
	}

	// Check parameters
	params->explorer_type = getInRange(params->explorer_type, 0, 3);
	params->scale_explorer_type = getInRange(params->scale_explorer_type, 0, 2);

	params->select_type = getInRange(params->select_type, 0, 2);
	params->select_overlap_type = getInRange(params->select_overlap_type, 0, 2);

	// Explorer - strategy for scanning the image (multiscale, pyramid, context ...)
	if ((m_explorer = Private::buildExplorer(params)) == 0)
	{
		return false;
	}

	// ScaleExplorer - fixed scale scanning method
	if ((m_scale_explorer = Private::buildScaleExplorer(params)) == 0)
	{
		return false;
	}

	// Scanner - main scanning object, contains the ROIs
	m_scanner = manage(new Scanner);
	m_scanner->deleteAllROIs();
	if (	m_scanner->setBOption("verbose", params->verbose) == false ||
		m_scanner->setExplorer(m_explorer) == false ||
		m_scanner->setSelector(Private::buildSelector(params)) == false)
	{
		return false;
	}

	// Set for each scale the feature extractors (<ipCore>s)
        //      [0/NULL] means the original image will be used as features!
	m_sp_prep = 0;
	if (params->prep_hlbp == true)
        {
        	if (m_sp_prep == 0)
        	{
        		m_sp_prep = manage(new spCoreChain);
        	}
        	m_sp_prep->add(manage(new ipLBPBitmap(manage(new ipLBP4R(1)))));
        }
        if (params->prep_ii == true)
	{
		if (m_sp_prep == 0)
        	{
        		m_sp_prep = manage(new spCoreChain);
        	}
        	m_sp_prep->add(manage(new ipIntegral));
	}

	// OK
	return true;
}

bool FaceFinder::reset(const char* filename)
{
	if (filename == 0)
	{
		return false;
	}

	FaceFinder::Params params;
	return params.load(filename) && reset(&params);
}

/////////////////////////////////////////////////////////////////////////
// Process some image to scan for patterns

bool FaceFinder::process(const Image& image)
{
	// Check parameters
	if (	m_scanner == 0 ||
		m_explorer == 0 ||
		m_scale_explorer == 0)
	{
		return false;
	}

	// Initialize processing
	if (m_scanner->init(image) == false)
	{
		return false;
	}

	if (	m_explorer->setScaleEvaluationIp(m_sp_prep) == false ||
		m_explorer->setScaleExplorer(m_scale_explorer) == false)
	{
		return false;
        }

        // Scan the image and get the results
        return m_scanner->process(image);
}

/////////////////////////////////////////////////////////////////////////
// Access functions

const PatternList& FaceFinder::getPatterns() const
{
	if (m_scanner == 0)
	{
		error("FaceFinder::getPatterns - invalid scanner!\n");
	}
	return m_scanner->getPatterns();
}

const Scanner& FaceFinder::getScanner() const
{
	if (m_scanner == 0)
	{
		error("FaceFinder::getScanner - invalid scanner!\n");
	}
	return *m_scanner;
}

Scanner& FaceFinder::getScanner()
{
	if (m_scanner == 0)
	{
		error("FaceFinder::getScanner - invalid scanner!\n");
	}
	return *m_scanner;
}

/////////////////////////////////////////////////////////////////////////

}
