#include "CmdLine.h"
#include "TensorFile.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* bindata_filename;
        char* tensor_filename;
	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Bindate to Tensor conversion program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("bindata file to read", &bindata_filename, "bindata file to read");
	cmd.addSCmdArg("tensor file to write", &tensor_filename, "tensor file to write");

	/* Possible options to add
		- type conversion from Float (default bindata) to any type
		- unfold 
	*/

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	//
	int n_examples = 0;
	int frame_size = 0;

	File file;

	file.open(bindata_filename, "r");
	file.read(&n_examples, sizeof(int), 1);
	file.read(&frame_size, sizeof(int), 1);

	//
	TensorFile tf;

	print("Writing tensor file ...\n");
	CHECK_FATAL(tf.openWrite(tensor_filename, Tensor::Float, 1, frame_size, 0, 0, 0));

	const TensorFile::Header& header = tf.getHeader();
	print("Tensor file:\n");
	print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
	//print(" n_tensors:    [%d]\n", header.m_n_samples);
	print(" n_tensors:    [%d]\n", n_examples);
	print(" n_dimensions: [%d]\n", header.m_n_dimensions);
	print(" size[0]:      [%d]\n", header.m_size[0]);
	print(" size[1]:      [%d]\n", header.m_size[1]);
	print(" size[2]:      [%d]\n", header.m_size[2]);
	print(" size[3]:      [%d]\n", header.m_size[3]);

	FloatTensor *tensor = new FloatTensor(frame_size);
	for(int i = 0 ; i < n_examples ; i++)
	{
		file.read(tensor->dataW(), sizeof(float), frame_size);

		if(verbose) tensor->sprint("Tensor %d", i);

		tf.save(*tensor);
	}

	delete tensor;



	tf.close();
	file.close();

        // OK
	return 0;
}

