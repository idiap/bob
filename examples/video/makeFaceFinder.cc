#include "torch5spro.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	FaceFinder::Params params;
	char* param_filename;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.addSCmdArg("Output parameter file", &param_filename, "File to write the scanning parameters");

	cmd.addText("\nModel options:");
	cmd.addSCmdArg("model", &params.f_model, "face classifier model");
	cmd.addDCmdOption("-model_thres", &params.f_thres, 0.0, "threshold of the face classifier model");
	cmd.addBCmdOption("-model_use_thres", &params.use_f_thres, false, "use the given threshold");

	cmd.addText("\nScanning options:");
	cmd.addICmdOption("-explorer_type", &params.explorer_type, 0, "explorer type: 0 - Pyramid, 1 - Multiscale, 2 - Context");
	cmd.addICmdOption("-scale_explorer_type", &params.scale_explorer_type, 0, "scale explorer type: 0 - Exhaustive, 1 - Spiral, 2 - Random");
	cmd.addICmdOption("-min_patt_w", &params.min_patt_w, 19, "minimum pattern width");
	cmd.addICmdOption("-max_patt_w", &params.max_patt_w, 190, "maximum pattern width");
	cmd.addICmdOption("-min_patt_h", &params.min_patt_h, 19, "minimum pattern height");
	cmd.addICmdOption("-max_patt_h", &params.max_patt_h, 190, "maximum pattern height");
	cmd.addDCmdOption("-dx", &params.dx, 0.2f, "Sub-window Oy position variation");
	cmd.addDCmdOption("-dy", &params.dy, 0.2f, "Sub-window Ox position variation");
	cmd.addDCmdOption("-ds", &params.ds, 1.25f, "Sub-window scale variation");
	cmd.addBCmdOption("-stop_at_first_detection", &params.stop_at_first_detection, false, "stop at first detection");
	cmd.addBCmdOption("-start_with_large_scale", &params.start_with_large_scales, false, "start with large scales");
	cmd.addICmdOption("-random_nsamples", &params.random_nsamples, 1024, "random scale explorer: number of samples");

	cmd.addText("\nPreprocessing options:");
	cmd.addBCmdOption("-prep_ii", &params.prep_ii, false, "Compute integral image");
	cmd.addBCmdOption("-prep_hlbp", &params.prep_hlbp, false, "HLBP: compute LBP4R bitmaps");

	cmd.addText("\nPruning options:");
	cmd.addBCmdOption("-prune_use_mean", &params.prune_use_mean, false, "prune using the mean");
	cmd.addBCmdOption("-prune_use_stdev", &params.prune_use_stdev, false, "prune using the stdev");
	cmd.addDCmdOption("-prune_min_mean", &params.prune_min_mean, 25.0, "prune using the mean: min value");
	cmd.addDCmdOption("-prune_max_mean", &params.prune_max_mean, 225.0, "prune using the mean: max value");
	cmd.addDCmdOption("-prune_min_stdev", &params.prune_min_stdev, 10.0, "prune using the stdev: min value");
	cmd.addDCmdOption("-prune_max_stdev", &params.prune_max_stdev, 125.0, "prune using the stdev: max value");

	cmd.addText("\nCandidate selection options:");
	cmd.addICmdOption("-select_type", &params.select_type, 1, "selector type: 0 - Overlap, 1 - MeanShift, 2 - No merge");
	cmd.addICmdOption("-select_overlap_type", &params.select_overlap_type, 0, "selector's merging type: 0 - Average, 1 - Confidence Weighted, 2 - Maximum Confidence");
	cmd.addBCmdOption("-select_overlap_iterative", &params.select_overlap_iterative, false, "Overlap: Iterative/One step");
	cmd.addICmdOption("-select_overlap_min_surf", &params.select_overlap_min_surf, 60, "Overlap: minimum surface overlap to merge");

	cmd.addText("\nContext-based model:");
	cmd.addSCmdOption("-context_model", &params.context_model, "", "Face context-based model");
	cmd.addICmdOption("-context_type", &params.context_type, 1, "Context type (0 - Full, 1 - Axis)");

        cmd.addText("\nGeneral options:");
	cmd.addBCmdOption("-verbose", &params.verbose, false, "verbose");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	// Save the parameters
	CHECK_FATAL(params.print());
	CHECK_FATAL(params.save(param_filename));

        // OK
	return 0;
}

