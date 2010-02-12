#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	double gamma = 0.2;
	double sigma0 = 1.0;
	double sigma1 = 2.0;
	int size = 2;
	double threshold = 10.;
	double alpha = 0.1;

	char* image_filename = 0;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for Tan and Triggs lighting normalization.\n");
	cmd.addSCmdArg("image", &image_filename, "input image");
	cmd.addDCmdOption("-gamma", &gamma, 0.2, "Gamma exponent of the gamma correction");
	cmd.addDCmdOption("-sigma0", &sigma0, 1.0, "sigma0 of the DoG filter (inner gaussian)");
	cmd.addDCmdOption("-sigma1", &sigma1, 2.0, "sigma1 of the DoG filter (outer gaussian)");
	cmd.addICmdOption("-size", &size, 2, "subsize of each side of the DoG filter (size=2*subsize+1)");
	cmd.addDCmdOption("-threshold", &threshold, 10.0, "Threshold for the contrast equalization");
	cmd.addDCmdOption("-alpha", &alpha, 0.1, "Exponent for defining a \"alpha\" norm");

	cmd.read(argc, argv);

	// Load the image
	Image image_(1, 1, 3);
	xtprobeImageFile xtprobe;
	CHECK_FATAL(xtprobe.load(image_, image_filename) == true);

	print("Processing image [width = %d, height = %d, nplanes = %d] ...\n",
		image_.size(1), image_.size(0), image_.size(2));

        // Convert it in Gray level (1 channel)
        Image image(image_.size(1),image_.size(0),1);
        CHECK_FATAL(image.copyFrom(image_) == true); 


	Image tanTriggs_image(1, 1, image.size(2));
	ipTanTriggs tanTriggs;

	// Perform Tan Triggs
	print("Performing Tan and Triggs...\n");
	CHECK_FATAL(tanTriggs.setDOption("gamma", gamma) == true);
	CHECK_FATAL(tanTriggs.setDOption("sigma0", sigma0) == true);
	CHECK_FATAL(tanTriggs.setDOption("sigma1", sigma1) == true);
	CHECK_FATAL(tanTriggs.setIOption("size", size) == true);
	CHECK_FATAL(tanTriggs.setDOption("threshold", threshold) == true);
	CHECK_FATAL(tanTriggs.setDOption("alpha", alpha) == true);
	CHECK_FATAL(tanTriggs.process(image) == true);
	CHECK_FATAL(tanTriggs.getNOutputs() == 1);
	CHECK_FATAL(tanTriggs.getOutput(0).getDatatype() == Tensor::Short);
	const ShortTensor& output = (const ShortTensor&)tanTriggs.getOutput(0);


	// Save the performed image
	char str[256];
	sprintf(str, "tanTriggs.pgm");
	CHECK_FATAL(tanTriggs_image.resize(output.size(1), output.size(0), output.size(2)) == true);
	CHECK_FATAL(tanTriggs_image.copyFrom(output) == true);
	CHECK_FATAL(xtprobe.save(tanTriggs_image, str) == true);


	Image histo_image(1, 1, image.size(2));

        // Perform Histogram Equalization
        ipHistoEqual histoEqual;
        print("Performing Histogram Equalization...\n");
        CHECK_FATAL(histoEqual.process(tanTriggs_image) == true);
        CHECK_FATAL(histoEqual.getNOutputs() == 1); 
        CHECK_FATAL(histoEqual.getOutput(0).getDatatype() == Tensor::Short);

        const ShortTensor& output_histo = (const ShortTensor&)histoEqual.getOutput(0);
        
	// Save the smoothed image
        CHECK_FATAL(histo_image.resize(output_histo.size(1), output_histo.size(0), output_histo.size(2)) == true);
	CHECK_FATAL(histo_image.copyFrom(output_histo) == true);
	CHECK_FATAL(xtprobe.save(histo_image, "tanTriggsHistoEqual.pgm") == true);

	return 0;
}

