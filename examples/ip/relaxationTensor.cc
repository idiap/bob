#include "torch5spro.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	char* input_filename = 0;
	char* output_filename = 0;
	int n_images;

	int steps = 20;
	int type = 1;
	double lambda = 1.;
	bool histo = false;

	// Read the command line
	CmdLine cmd;
	cmd.info("Perform Relaxation Algorithm over a TensorFile and save it to a TensorFile.\n");
	cmd.addSCmdArg("inputTensorFile", &input_filename, "input tensor filename");
	cmd.addSCmdArg("outputTensorFile", &output_filename, "output tensor filename");
	cmd.addICmdOption("-I", &n_images, -1, "number of images to process");
	cmd.addICmdOption("-steps", &steps, 20, "Number of relaxation steps");
	cmd.addICmdOption("-type", &type, 1, "Type of diffusion: 0 = Isotropic, 1 = Anisotropic (Weber contrast)");
	cmd.addDCmdOption("-lambda", &lambda, 1., "Relative importance of the smoothness constraint");
	cmd.addBCmdOption("-histo", &histo, false, "Perform an histogram equalization after processing each images");

	if (cmd.read(argc, argv) < 0)
	{
		message("Error parsing the command line");
		return 0;
	}



	// Load input tensor
	TensorFile itf;
	CHECK_FATAL( itf.openRead( input_filename ) );

	print("Reading tensor header file ...\n");
	const TensorFile::Header& header = itf.getHeader();

	print("Tensor file:\n");
	print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
	print(" n_tensors:    [%d]\n", header.m_n_samples);
	print(" n_dimensions: [%d]\n", header.m_n_dimensions);
	print(" size[0]:      [%d]\n", header.m_size[0]);
	print(" size[1]:      [%d]\n", header.m_size[1]);
	print(" size[2]:      [%d]\n", header.m_size[2]);
	print(" size[3]:      [%d]\n", header.m_size[3]);

	if(n_images == -1) n_images = header.m_n_samples;

	if(header.m_type != Tensor::Short)
	{
		warning("Unsupported tensor type (Short only).");

		return 1;
	}

	if(header.m_n_dimensions != 2)
	{
		warning("Unsupported dimensions (2 only).");

		return 1;
	}

	TensorFile otf;
	CHECK_FATAL(otf.openWrite( output_filename, header.m_type, header.m_n_dimensions, header.m_size[0], header.m_size[1], header.m_size[2], header.m_size[3]));

	print("Reading tensor header file ...\n");
	const TensorFile::Header& header2 = otf.getHeader();

	print("Tensor file:\n");
	print(" type:         [%s]\n", str_TensorTypeName[header2.m_type]);
	print(" n_tensors:    [%d]\n", header2.m_n_samples);
	print(" n_dimensions: [%d]\n", header2.m_n_dimensions);
	print(" size[0]:      [%d]\n", header2.m_size[0]);
	print(" size[1]:      [%d]\n", header2.m_size[1]);
	print(" size[2]:      [%d]\n", header2.m_size[2]);
	print(" size[3]:      [%d]\n", header2.m_size[3]);
	//
	Tensor *tensor = NULL;
	ipRelaxation relaxation;

	CHECK_FATAL(relaxation.setIOption("steps", steps) == true);
	CHECK_FATAL(relaxation.setIOption("type", type) == true);
	CHECK_FATAL(relaxation.setDOption("lambda", lambda) == true);

	ipHistoEqual histoEqual;

	// Loop over each Tensor contained in the TensorFile
	print("\nPerforming Relaxation Algorithm over each sample ...\n");
	ShortTensor* out_2d;
	while ((tensor = itf.load()) != 0)
	{
		Image imagegray(tensor->size(1), tensor->size(0), 1);
		ShortTensor *t_ = new ShortTensor();
		t_->select(&imagegray, 2, 0);
		t_->copy(tensor);
		delete t_;


		CHECK_FATAL(relaxation.process(imagegray) == true);
		CHECK_FATAL(relaxation.getNOutputs() == 1);
		CHECK_FATAL(relaxation.getOutput(0).getDatatype() == Tensor::Short);

		if (histo)
		{
			CHECK_FATAL(histoEqual.process(relaxation.getOutput(0)) == true);
			CHECK_FATAL(histoEqual.getNOutputs() == 1);
			CHECK_FATAL(histoEqual.getOutput(0).getDatatype() == Tensor::Short);

			// Remove the third dimension and save the result to a file
			out_2d=new ShortTensor();
			out_2d->select( &((const ShortTensor&)histoEqual.getOutput(0)), 2, 0 );
		}
		else
		{
			// Remove the third dimension and save the result to a file
			ShortTensor* out_2d=new ShortTensor();
			out_2d->select( &((const ShortTensor&)relaxation.getOutput(0)), 2, 0 );
		}

		CHECK_FATAL( otf.save( *out_2d ));

		delete out_2d;	
		delete tensor;
	}

	itf.close();
	otf.close();

	//
	print("OK\n");

	return 0;
}

