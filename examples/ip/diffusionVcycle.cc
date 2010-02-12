#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* image_filename = 0;
	int n_grids = 5;
	int type = 1;
	double lambda = 1.;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for Diffusion V-Cycle.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");
	cmd.addICmdOption("-n_grids", &n_grids, 5, "number of levels (WARNING: dependant on input image)");
	cmd.addICmdOption("-type", &type, 1, "Type of diffusion: 0 = Isotropic, 1 = Anisotropic (Weber contrast)");
	cmd.addDCmdOption("-lambda", &lambda, 0.5, "Relative importance of the smoothness constraint");

	cmd.read(argc, argv);

	// Load the image to play with (Gray or RGB)
	xtprobeImageFile xtprobe;
	Image image_(1, 1, 3);

	CHECK_FATAL(xtprobe.load(image_, image_filename) == true);
	print("Loaded image of size [%dx%d] with [%d] planes.\n\n",
		image_.getWidth(), image_.getHeight(), image_.getNPlanes());

	// Convert it in Gray level (1 channel)
	Image image(image_.size(1),image_.size(0),1);
	CHECK_FATAL(image.copyFrom(image_) == true);

	Image vcycle_image(1, 1, image.size(2));

	// Process image
	ipVcycle vcycle;
	CHECK_FATAL(vcycle.setIOption("n_grids", n_grids) == true);
	CHECK_FATAL(vcycle.setIOption("type", type) == true);
	CHECK_FATAL(vcycle.setDOption("lambda", lambda) == true);
	CHECK_FATAL(vcycle.process(image) == true);
	CHECK_FATAL(vcycle.getNOutputs() == 1);
	CHECK_FATAL(vcycle.getOutput(0).getDatatype() == Tensor::Short);
	
	const ShortTensor& output_vcycle = (const ShortTensor&)vcycle.getOutput(0);
        CHECK_FATAL(vcycle_image.resize(output_vcycle.size(1), output_vcycle.size(0), output_vcycle.size(2)) == true);
	CHECK_FATAL(vcycle_image.copyFrom(output_vcycle) == true);
	CHECK_FATAL(xtprobe.save(vcycle_image, "vcycleImage.pgm") == true);

	Image histo_image(1, 1, image.size(2));

        // Perform Histogram Equalization
        ipHistoEqual histoEqual;
        print("Performing Histogram Equalization...\n");
        CHECK_FATAL(histoEqual.process(vcycle_image) == true);
        CHECK_FATAL(histoEqual.getNOutputs() == 1); 
        CHECK_FATAL(histoEqual.getOutput(0).getDatatype() == Tensor::Short);

        const ShortTensor& output_histo = (const ShortTensor&)histoEqual.getOutput(0);
        
	// Save the smoothed image
        CHECK_FATAL(histo_image.resize(output_histo.size(1), output_histo.size(0), output_histo.size(2)) == true);
	CHECK_FATAL(histo_image.copyFrom(output_histo) == true);
	CHECK_FATAL(xtprobe.save(histo_image, "vcycleImageHistoEqual.pgm") == true);


	return 0;
}

