#include "CmdLine.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "TensorFile.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* tensor_filename;
        char* image_extension;
        char* image_basename;
	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Tensor read program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("tensor file to test", &tensor_filename, "tensor file to read");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
	cmd.addSCmdOption("-xt", &image_extension, "pgm", "image extension");
	cmd.addSCmdOption("-base", &image_basename, "tensor", "image basename");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	TensorFile tf;

	CHECK_FATAL(tf.openRead(tensor_filename));

	print("Reading tensor header file ...\n");
	const TensorFile::Header& header = tf.getHeader();

	print("Tensor file:\n");
	print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
	print(" n_tensors:    [%d]\n", header.m_n_samples);
	print(" n_dimensions: [%d]\n", header.m_n_dimensions);
	print(" size[0]:      [%d]\n", header.m_size[0]);
	print(" size[1]:      [%d]\n", header.m_size[1]);
	print(" size[2]:      [%d]\n", header.m_size[2]);
	print(" size[3]:      [%d]\n", header.m_size[3]);

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

	//
	Tensor *tensor = NULL;
	int i = 0;
	char image_filename[250];
	
	tensor = tf.load();
	if(verbose) tensor->sprint("%d", i);
	Image imagegray(tensor->size(1), tensor->size(0), 1);
	ShortTensor *t_ = new ShortTensor();
	t_->select(&imagegray, 2, 0);
	t_->copy(tensor);
	xtprobeImageFile xtprobe;
	sprintf(image_filename, "%s%05d.%s", image_basename, i, image_extension);
	xtprobe.save(imagegray, image_filename);
	delete t_;
	delete tensor;
	i++;
	while ((tensor = tf.load()) != 0)
	{
		if(verbose) tensor->sprint("%d", i);
		Image imagegray(tensor->size(1), tensor->size(0), 1);
		ShortTensor *t_ = new ShortTensor();
		t_->select(&imagegray, 2, 0);
		t_->copy(tensor);
		xtprobeImageFile xtprobe;
		sprintf(image_filename, "%s%05d.%s", image_basename, i, image_extension);
		xtprobe.save(imagegray, image_filename);
		delete t_;
		delete tensor;
		i++;
	}

	tf.close();

        // OK
	return 0;
}

