#include "ipBlock.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "CmdLine.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	bool save;
	int n_tests;

	// Read the command line
	CmdLine cmd;
	cmd.info("Testing program for block decomposition.\n");
	cmd.addText("\nOptions:");
	cmd.addBCmdOption("-save", &save, false, "save");
	cmd.addICmdOption("-t", &n_tests, 1, "number of tests");

	cmd.read(argc, argv);

	xtprobeImageFile xtprobe;

	// Load the image to play with
	const char* image1filename = "../data/images/1001_f_g1_s01_1001_en_1.jpeg";
	Image image1(1, 1, 3);

	CHECK_FATAL(xtprobe.load(image1, image1filename) == true);
	print("Loaded image 1 of size [%dx%d] with [%d] planes.\n\n",
		image1.getWidth(), image1.getHeight(), image1.getNPlanes());

	const char* image2filename = "../data/images/003_1_1.pgm";
	Image image2(1, 1, 1);

	CHECK_FATAL(xtprobe.load(image2, image2filename) == true);
	print("Loaded image 2 of size [%dx%d] with [%d] planes.\n\n",
		image2.getWidth(), image2.getHeight(), image2.getNPlanes());

	ipBlock ipblock;
	Image block_image;

	// Crop some random parts
	srand((unsigned int)time(0));

	for (int t = 0; t < n_tests; t ++)
	{
	   	//
	   	// Image 1: color image
		int block_w = rand() % image1.getWidth();
		int block_h = rand() % image1.getHeight();
		int ox = rand() % block_w;
		int oy = rand() % block_h;

		print("[%d/%d]: Image 1 blocks [%d, %d, %d, %d] ...\n",
			t + 1, n_tests, block_w, block_h, ox, oy);
		CHECK_FATAL(ipblock.setIOption("ox", ox) == true);
		CHECK_FATAL(ipblock.setIOption("oy", oy) == true);
		CHECK_FATAL(ipblock.setIOption("w", block_w) == true);
		CHECK_FATAL(ipblock.setIOption("h", block_h) == true);
		CHECK_FATAL(ipblock.process(image1) == true);
		
		print("Number of output blocks: %d\n", ipblock.getNOutputs());

		// Save blocks to images
		block_image.resize(block_w, block_h, image1.getNPlanes());
		for(int b = 0 ; b < ipblock.getNOutputs() ; b++)
		{
			block_image.copyFrom(ipblock.getOutput(b));

			if(save)
			{
				char str[200];
				sprintf(str, "image1_block_%d_%d.jpg", t, b);
				CHECK_FATAL(xtprobe.save(block_image, str) == true);
			}
		}

		//
		// Image 2: grayscale image
		block_w = rand() % image2.getWidth();
		block_h = rand() % image2.getHeight();
		ox = rand() % block_w;
		oy = rand() % block_h;

		print("[%d/%d]: Image 2 blocks [%d, %d, %d, %d] ...\n",
			t + 1, n_tests, block_w, block_h, ox, oy);
		CHECK_FATAL(ipblock.setIOption("ox", ox) == true);
		CHECK_FATAL(ipblock.setIOption("oy", oy) == true);
		CHECK_FATAL(ipblock.setIOption("w", block_w) == true);
		CHECK_FATAL(ipblock.setIOption("h", block_h) == true);
		CHECK_FATAL(ipblock.process(image2) == true);
		
		print("Number of output blocks: %d\n", ipblock.getNOutputs());

		// Save blocks to images
		block_image.resize(block_w, block_h, image2.getNPlanes());
		for(int b = 0 ; b < ipblock.getNOutputs() ; b++)
		{
			block_image.copyFrom(ipblock.getOutput(b));

			if(save)
			{
				char str[200];
				sprintf(str, "image2_block_%d_%d.jpg", t, b);
				CHECK_FATAL(xtprobe.save(block_image, str) == true);
			}
		}

		//
		// Image 2: grayscale image -> 4D tensor

		print("[%d/%d]: Image 2 blocks [%d, %d, %d, %d] ...\n",
			t + 1, n_tests, block_w, block_h, ox, oy);
		CHECK_FATAL(ipblock.setBOption("rcoutput", true) == true);
		CHECK_FATAL(ipblock.setIOption("ox", ox) == true);
		CHECK_FATAL(ipblock.setIOption("oy", oy) == true);
		CHECK_FATAL(ipblock.setIOption("w", block_w) == true);
		CHECK_FATAL(ipblock.setIOption("h", block_h) == true);
		CHECK_FATAL(ipblock.process(image2) == true);
		
		print("Number of output blocks: %d\n", ipblock.getNOutputs());

		ShortTensor &t_rcoutput = (ShortTensor &) ipblock.getOutput(0);
		
		print("   n_dimensions: [%d]\n", t_rcoutput.nDimension());
		print("   size[0]:      [%d]\n", t_rcoutput.size(0));
		print("   size[1]:      [%d]\n", t_rcoutput.size(1));
		print("   size[2]:      [%d]\n", t_rcoutput.size(2));
		print("   size[3]:      [%d]\n", t_rcoutput.size(3));

		CHECK_FATAL(t_rcoutput.size(2) == block_h);
		CHECK_FATAL(t_rcoutput.size(3) == block_w);
		
		int n_rows = t_rcoutput.size(0);
		int n_cols = t_rcoutput.size(1);

		// Save blocks to images
		block_image.resize(block_w, block_h, image2.getNPlanes());

		ShortTensor *t_rcoutput_narrow_rows = new ShortTensor();
		ShortTensor *t_rcoutput_narrow_cols = new ShortTensor();
		ShortTensor *t_block = new ShortTensor();

		t_block->select(&block_image, 2, 0);
		
		for(int r = 0; r < n_rows; r++)
		{
			//t_rcoutput_narrow_rows->narrow(&t_rcoutput, 0, r, 1);
			t_rcoutput_narrow_rows->select(&t_rcoutput, 0, r);

		   	for(int c = 0; c < n_cols; c++) 
			{
				//t_rcoutput_narrow_cols->narrow(t_rcoutput_narrow_rows, 1, c, 1);
				t_rcoutput_narrow_cols->select(t_rcoutput_narrow_rows, 0, c);
				
				t_block->copy(t_rcoutput_narrow_cols);

				if(save)
				{
					char str[200];
					sprintf(str, "image2_block_%d_%dx%d.jpg", t, r, c);
					CHECK_FATAL(xtprobe.save(block_image, str) == true);
				}
			}
		}
	
		delete t_block;
		delete t_rcoutput_narrow_cols;
		delete t_rcoutput_narrow_rows;

		// switch it off after use
		CHECK_FATAL(ipblock.setBOption("rcoutput", false) == true);
	}

	print("\nOK\n");

	return 0;
}

