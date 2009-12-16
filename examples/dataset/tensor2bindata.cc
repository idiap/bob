#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* tensor_filename;
        char* bindata_filename;
	bool verbose;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Tensor to Bindata conversion");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("tensor file to test", &tensor_filename, "tensor file to read");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
	cmd.addSCmdOption("-b", &bindata_filename, "output.bindata", "bindata filename");

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

	// get the file for bindata ready
 	File binfile;

	binfile.open(bindata_filename, "w");
	binfile.write(&header.m_n_samples, sizeof(int), 1);
	int frame_size = header.m_size[0]*header.m_size[1];
	binfile.write(&frame_size, sizeof(int), 1);

 	float *data = new float[frame_size];

	//
	ShortTensor *tensor = NULL;
	int i = 0;

	tensor = (ShortTensor *)tf.load();

	if(verbose) tensor->sprint("%d", i);
	for(int m=0;m<header.m_size[0];m++)
        	for(int n=0;n<header.m_size[1];n++)
        	{
            		data[m*header.m_size[1]+n] = ((*tensor)(m,n)+0.0)/255.0;
        	}
        binfile.write(data, sizeof(float), frame_size);

	delete tensor;
	i++;
	while ((tensor = (ShortTensor*)tf.load()) != 0)
	{
		if(verbose) tensor->sprint("%d", i);
		for(int m=0;m<header.m_size[0];m++)
        		for(int n=0;n<header.m_size[1];n++)
        		{
            			data[m*header.m_size[1]+n] = ((*tensor)(m,n)+0.0)/255.0;
        		}
        	binfile.write(data, sizeof(float), frame_size);
		i++;
	}

	tf.close();
	binfile.close();

        // OK
	return 0;
}

