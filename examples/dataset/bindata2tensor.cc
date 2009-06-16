#include "torch5spro.h"

using namespace Torch;


int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* bindata_filename;
        char* tensor_filename;
	float scalar;
	bool verbose;
	int ttype_out;
	int unfold;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Bindate to Tensor conversion program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("bindata file to read", &bindata_filename, "bindata file to read");
	cmd.addSCmdArg("tensor file to write", &tensor_filename, "tensor file to write");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "verbose");
	cmd.addFCmdOption("-s", &scalar, 1.0, "scale the data");
	cmd.addICmdOption("-tt", &ttype_out, 4, "output tensor type (0=Char, 1=Short, 2=Int, 3=Long, 4=Float, 5=Double)");
	cmd.addICmdOption("-unfold", &unfold, -1, "unfold argument");

	// Parse the command line
	if (cmd.read(argc, argv) < 0)
	{
		return 0;
	}

	if(ttype_out < 0 || ttype_out > 5)
	{
	   	warning("incorrect output tensor type.");
		return 0;
	}

	Tensor::Type ttype = (Tensor::Type) ttype_out;

	//
	int n_examples = 0;
	int frame_size = 0;

	File file;

	file.open(bindata_filename, "r");
	file.read(&n_examples, sizeof(int), 1);
	file.read(&frame_size, sizeof(int), 1);

	//
	TensorFile tf;

	int width = 0;
	int height = 0;

	if(unfold > 1)
	{
	   	width = unfold;
		height = frame_size / width;

		print("Writing unfolded (%dx%d) tensor file ...\n", height, width);

		CHECK_FATAL(tf.openWrite(tensor_filename, ttype, 2, height, width, 0, 0));
	}
	else
	{
		print("Writing tensor file ...\n");

		//CHECK_FATAL(tf.openWrite(tensor_filename, Tensor::Float, 1, frame_size, 0, 0, 0));
		CHECK_FATAL(tf.openWrite(tensor_filename, ttype, 1, frame_size, 0, 0, 0));
	}

	const TensorFile::Header& header = tf.getHeader();
	print("Tensor file:\n");
	print(" type:         [%s]\n", str_TensorTypeName[header.m_type]);
	print(" n_tensors:    [%d]\n", n_examples);
	print(" n_dimensions: [%d]\n", header.m_n_dimensions);
	print(" size[0]:      [%d]\n", header.m_size[0]);
	print(" size[1]:      [%d]\n", header.m_size[1]);
	print(" size[2]:      [%d]\n", header.m_size[2]);
	print(" size[3]:      [%d]\n", header.m_size[3]);

	FloatTensor *tensor = new FloatTensor(frame_size);
	FloatTensor *unfoldtensor = new FloatTensor;
	Tensor *otensor = NULL;
	switch (ttype)
	{
	case Tensor::Char:
		if(unfold > 1) otensor = new CharTensor(height, width);
		else otensor = new CharTensor(frame_size);

		break;

	case Tensor::Short:
		if(unfold > 1) otensor = new ShortTensor(height, width);
		else otensor = new ShortTensor(frame_size);

		break;

	case Tensor::Int:
		if(unfold > 1) otensor = new IntTensor(height, width);
		else otensor = new IntTensor(frame_size);

		break;

	case Tensor::Long:
		if(unfold > 1) otensor = new LongTensor(height, width);
		else otensor = new LongTensor(frame_size);

		break;

	case Tensor::Float:
		if(unfold > 1) otensor = new FloatTensor(height, width);
		else otensor = new FloatTensor(frame_size);

		break;

	case Tensor::Double:
		if(unfold > 1) otensor = new DoubleTensor(height, width);
		else otensor = new DoubleTensor(frame_size);

		break;
	}

	for(int i = 0 ; i < n_examples ; i++)
	{
		file.read(tensor->dataW(), sizeof(float), frame_size);

		THFloatTensor_mul(tensor->t, scalar);

		if(verbose) tensor->sprint("%d", i);

		if(unfold > 1)
		{
			unfoldtensor->unfold(tensor, 0, width, width);

			otensor->copy(unfoldtensor);
		}
		else otensor->copy(tensor);

		tf.save(*otensor);
	}

	delete unfoldtensor;
	delete otensor;
	delete tensor;

	tf.close();
	file.close();

        // OK
	return 0;
}

