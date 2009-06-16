#include "torch5spro.h"

using namespace Torch;

//////////////////////////////////////////////////////////////////////////////////////
// Checks some tensor file
//////////////////////////////////////////////////////////////////////////////////////

bool checkTensorFile(	TensorFile& tf, int n_tensors,
			Tensor::Type ttype,
			int tdims, int tsize0, int tsize1, int tsize2, int tsize3)
{
	const TensorFile::Header& header = tf.getHeader();

	print(" >>> checking tensor file with [%d] samples ...\n", header.m_n_samples);

	CHECK_FATAL(header.m_n_samples == n_tensors);
	CHECK_FATAL(header.m_type == ttype);
	CHECK_FATAL(header.m_n_dimensions == tdims);
	CHECK_FATAL(header.m_size[0] == tsize0);
	CHECK_FATAL(header.m_size[1] == tsize1);
	CHECK_FATAL(header.m_size[2] == tsize2);
	CHECK_FATAL(header.m_size[3] == tsize3);

	// Sequential reading
	print(" >>> sequential reading ...\n");
	Tensor* tensor = 0;
	int cnt = 0;
	while ((tensor = tf.load()) != 0)
	{
		CHECK_FATAL(tensor != 0);
		CHECK_FATAL(((const ShortTensor*)tensor)->get(0, 0, 0) == (cnt ++));
		delete tensor;
	}

	// Random reading
	if (header.m_n_samples > 0)
	{
		print(" >>> random reading ...\n");
		tensor = tf.load(0);
		CHECK_FATAL(tensor != 0);

		for (int j = 0; j < header.m_n_samples; j ++)
		{
			const int index = rand() % header.m_n_samples;
			CHECK_FATAL(tf.load(*tensor, index) == true);
			CHECK_FATAL(((const ShortTensor*)tensor)->get(0, 0, 0) == index);
		}
		delete tensor;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////////////////
// Main
//////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char* argv[])
{
	const int n_tests = 50;
	const int n_max_tensors = 100;

	srand((unsigned int)time(0));

	const char* tfilename = "test.tensor";
	const Tensor::Type ttype = Tensor::Short;
	const int tdims = 3;
	const int tsize0 = 320;
	const int tsize1 = 240;
	const int tsize2 = 3;
	const int tsize3 = 0;

	ShortTensor tensor(tsize0, tsize1, tsize2);

	// Open the file for writing - this will clear the file
	TensorFile tf;
	CHECK_FATAL(tf.openWrite(tfilename, ttype, tdims, tsize0, tsize1, tsize2, tsize3));
	tf.close();

	// Do some random appends, writes and reads
	int n_tensors = 0;
	for (int i = 0; i < n_tests; i ++)
	{
		print("[%d/%d]", i + 1, n_tests);
		switch (rand() % 3)
		{
		case 0:	// Read some tensors
			{
				print("Reading %d tensors ...\n", n_tensors);

				CHECK_FATAL(tf.openRead(tfilename));
				checkTensorFile(tf, n_tensors, ttype, tdims, tsize0, tsize1, tsize2, tsize3);
				tf.close();
			}
			break;

		case 1:	// Append some tensors
			{
				CHECK_FATAL(tf.openAppend(tfilename));
				const int n_appends = rand() % n_max_tensors;
				for (int j = 0; j < n_appends; j ++)
				{
					tensor.fill(n_tensors ++);
					CHECK_FATAL(tf.save(tensor));
				}
				tf.close();

				print("Appending %d tensors ...\n", n_appends);
			}
			break;

		case 2:	// Write some tensors
			{
				CHECK_FATAL(tf.openWrite(tfilename, ttype, tdims, tsize0, tsize1, tsize2, tsize3));
				const int n_writes = rand() % n_max_tensors;
				n_tensors = 0;
				for (int j = 0; j < n_writes; j ++)
				{
					tensor.fill(n_tensors ++);
					CHECK_FATAL(tf.save(tensor));
				}
				tf.close();

				print("Writing %d tensors ...\n", n_writes);
			}
			break;
		}
	}

   	return 0;
}

