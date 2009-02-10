#include "CmdLine.h"
#include "TensorFile.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* tensor_filename;
	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Tensor read program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("tensor file to test", &tensor_filename, "tensor file to read");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");

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

	Tensor *tensor = NULL;
	int i = 0;
	tensor = tf.load();
	if(verbose) tensor->sprint("Tensor %d:", i);
	delete tensor;
	i++;
	while ((tensor = tf.load()) != 0)
	{
		if(verbose) tensor->sprint("Tensor %d:", i);
		delete tensor;
		i++;
	}
	tf.close();

        // OK
	return 0;
}

