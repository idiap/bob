#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* image_filename = 0;
	int steps = 20;
	int type = 1;
	double lambda = 1.;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for Relaxation.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");
	cmd.addICmdOption("-steps", &steps, 20, "Number of relaxation steps");
	cmd.addICmdOption("-type", &type, 1, "Type of diffusion: 0 = Isotropic, 1 = Anisotropic (Weber contrast)");
	cmd.addDCmdOption("-lambda", &lambda, 1., "Relative importance of the smoothness constraint");

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

	Image relaxation_image(1, 1, image.size(2));

	// Process image
	ipRelaxation relaxation;
	CHECK_FATAL(relaxation.setIOption("steps", steps) == true);
	CHECK_FATAL(relaxation.setIOption("type", type) == true);
	CHECK_FATAL(relaxation.setDOption("lambda", lambda) == true);
	CHECK_FATAL(relaxation.process(image) == true);
	CHECK_FATAL(relaxation.getNOutputs() == 1);
	CHECK_FATAL(relaxation.getOutput(0).getDatatype() == Tensor::Short);
	
	const ShortTensor& output_relaxation = (const ShortTensor&)relaxation.getOutput(0);
        CHECK_FATAL(relaxation_image.resize(output_relaxation.size(1), output_relaxation.size(0), output_relaxation.size(2)) == true);
	CHECK_FATAL(relaxation_image.copyFrom(output_relaxation) == true);
	CHECK_FATAL(xtprobe.save(relaxation_image, "relaxationImage.pgm") == true);

	Image histo_image(1, 1, image.size(2));

        // Perform Histogram Equalization
        ipHistoEqual histoEqual;
        print("Performing Histogram Equalization...\n");
        CHECK_FATAL(histoEqual.process(relaxation_image) == true);
        CHECK_FATAL(histoEqual.getNOutputs() == 1); 
        CHECK_FATAL(histoEqual.getOutput(0).getDatatype() == Tensor::Short);

        const ShortTensor& output_histo = (const ShortTensor&)histoEqual.getOutput(0);
        
	// Save the smoothed image
        CHECK_FATAL(histo_image.resize(output_histo.size(1), output_histo.size(0), output_histo.size(2)) == true);
	CHECK_FATAL(histo_image.copyFrom(output_histo) == true);
	CHECK_FATAL(xtprobe.save(histo_image, "relaxationImageHistoEqual.pgm") == true);

	//CHECK_FATAL(msr_image.resize(output.size(1), output.size(0), output.size(2)) == true);
	//CHECK_FATAL(xtprobe.save(msr_image, "multiscaleRetinex.jpg") == true);

	print("\nOK\n");

	return 0;
}

