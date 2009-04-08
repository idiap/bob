#include "Image.h"
#include "xtprobeImageFile.h"
#include "TensorFile.h"
#include "spDCT.h"
#include "ipBlock.h"
#include "CmdLine.h"

using namespace Torch;

struct zigzagDCT
{
	int x;
	int y;
};

void compute_zigzag(int block_size_h, int block_size_w, zigzagDCT *index);

void retainDC(const FloatTensor *dct2d_in, int n_dc, FloatTensor *dc_out, zigzagDCT *index);

int main(int argc, char* argv[])
{
        ///////////////////////////////////////////////////////////////////
        // Parse the command line
        ///////////////////////////////////////////////////////////////////

	// Set options
        char* tensor_filename;
        char* output_basename;
	bool verbose;
	int block_size_h;
	int block_size_w;
	int block_overlap_h;
	int block_overlap_w;
	int n_dc;

	// Build the command line object
	CmdLine cmd;
	cmd.setBOption("write log", false);

	cmd.info("Tensor read program");

	cmd.addText("\nArguments:");
	cmd.addSCmdArg("tensor file to test", &tensor_filename, "tensor file to read");

	cmd.addText("\nBlock decomposition:");
	cmd.addICmdOption("-sizeH", &block_size_h, 8, "block size H");
	cmd.addICmdOption("-sizeW", &block_size_w, 8, "block size W");
	cmd.addICmdOption("-overlapH", &block_overlap_h, 4, "overlap H between blocks");
	cmd.addICmdOption("-overlapW", &block_overlap_w, 4, "overlap W between blocks");

	cmd.addText("\nDCT:");
	cmd.addICmdOption("-dc", &n_dc, 15, "number of DC coefficients to retain");

	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-verbose", &verbose, false, "print Tensor values");
	cmd.addSCmdOption("-o", &output_basename, "dct", "basename");

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
	if(n_dc > block_size_h * block_size_w)
	{
		warning("Impossible to retain more than %d DC.", block_size_h * block_size_w);

		return 1;
	}

	//
	spDCT dct;

	zigzagDCT *dc_zigzag_index = new zigzagDCT [block_size_h * block_size_w];

	compute_zigzag(block_size_h, block_size_w, dc_zigzag_index);

	//
	ipBlock ipblock;

	ipblock.setBOption("verbose", verbose);
	ipblock.setBOption("rcoutput", true);
	ipblock.setIOption("ox", block_overlap_w);
	ipblock.setIOption("oy", block_overlap_h);
	ipblock.setIOption("w", block_size_w);
	ipblock.setIOption("h", block_size_h);
		
	//
	Tensor *tensor = NULL;
	char ofilename[250];

	TensorFile *ofile = new TensorFile;
	sprintf(ofilename, "%s.tensor", output_basename);
	//ofile->openWrite(ofilename, Tensor::Short, 2, block_size_h, block_size_w, 0, 0);
	ofile->openWrite(ofilename, Tensor::Float, 1, n_dc, 0, 0, 0);

	for(int t = 0 ; t < header.m_n_samples ; t++)
	{
		tensor = tf.load();

		Image imagegray(tensor->size(1), tensor->size(0), 1);
		ShortTensor *t_ = new ShortTensor();
		t_->select(&imagegray, 2, 0);
		t_->copy(tensor);

		ipblock.process(imagegray);

		if(verbose) print("Number of output blocks: %d\n", ipblock.getNOutputs());

		ShortTensor &t_rcoutput = (ShortTensor &) ipblock.getOutput(0);
		int n_rows = t_rcoutput.size(0);
		int n_cols = t_rcoutput.size(1);

		ShortTensor *t_rcoutput_narrow_rows = new ShortTensor();
		ShortTensor *t_rcoutput_narrow_cols = new ShortTensor();
		ShortTensor *t_block = new ShortTensor(block_size_h, block_size_w);
		FloatTensor *t_dc = new FloatTensor(n_dc);

		for(int r = 0; r < n_rows; r++)
		{
			//t_rcoutput_narrow_rows->narrow(&t_rcoutput, 0, r, 1);
			t_rcoutput_narrow_rows->select(&t_rcoutput, 0, r);

		   	for(int c = 0; c < n_cols; c++) 
			{
				//t_rcoutput_narrow_cols->narrow(t_rcoutput_narrow_rows, 1, c, 1);
				t_rcoutput_narrow_cols->select(t_rcoutput_narrow_rows, 0, c);

				t_block->copy(t_rcoutput_narrow_cols);

				dct.process(*t_block);
				
				const FloatTensor& out = (const FloatTensor&) dct.getOutput(0);

				retainDC(&out, n_dc, t_dc, dc_zigzag_index);
				
				ofile->save(*t_dc);
			}
		}

		delete t_dc;
		delete t_block;
		delete t_rcoutput_narrow_cols;
		delete t_rcoutput_narrow_rows;

		delete t_;
		delete tensor;
	}

	tf.close();

	delete ofile;

	delete [] dc_zigzag_index;


        // OK
	return 0;
}

void retainDC(const FloatTensor *dct2d_in, int n_dc, FloatTensor *dc_out, zigzagDCT *index)
{
	//dct2d_in->print("DCT 2D");
	
	for(int i = 0 ; i < n_dc ; i++) 
	{
		float z = dct2d_in->get(index[i].y, index[i].x);
		dc_out->set(i, z);
	}

	//dc_out->print("DC retained");
}


void compute_zigzag(int block_size_h, int block_size_w, zigzagDCT *index)
{
   	print("Computing zig-zag pattern for DCT ...\n");
	
	int x = 0, y = 0;
	int id = 0;

	index[id].x = x;
	index[id++].y = y;
	
	for(;;)
	{
		/* right or down ?  */
		if(x != block_size_w-1) x = x + 1;
		else y = y + 1;
		
		index[id].x = x;
		index[id++].y = y;
		
		/* now down left..  */
		while(x != 0 && y != block_size_h-1)
		{
			x = x - 1;
			y = y + 1;
			index[id].x = x;
			index[id++].y = y;
		}

		/* right or down ? */
		if(y == block_size_h-1)
		{
			x = x + 1;
			index[id].x = x;
			index[id++].y = y;
			if(x == block_size_w-1) break;
		}
		else
		{
			y = y + 1;
			index[id].x = x;
			index[id++].y = y;
		}

		/* finally up right..  */
		while(x != block_size_w-1 && y != 0)
		{
			x = x + 1;
			y = y - 1;
			index[id].x = x;
			index[id++].y = y;
		}
	}

	ShortTensor t_zigzag(block_size_h, block_size_w);

	//print("Zig-Zag pattern:\n");
	for(int i = 0 ; i < block_size_h * block_size_w ; i++) 
	{
		//print("DCT %d: x=%d y=%d\n", i, index[i].x, index[i].y);
		t_zigzag(index[i].y, index[i].x) = i+1;
	}
	t_zigzag.print("Zig-Zag Matrix");
}

