#include "File.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "Machines.h"

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
        // Check arguments
        if (argc != 3)
        {
                print("\nUsage: <test19x19models> <model filename> <bindata file to test>\n\n");
                return 1;
        }
        const char* model_filename = argv[1];
        const char* data_filename = argv[2];

        Image image;

	// Load the cascade machine
	CascadeMachine* cascade = (CascadeMachine*)Torch::loadMachineFromFile(model_filename);
	if (cascade == 0)
	{
		print("ERROR: loading model [%s]!\n", model_filename);
		return 1;
	}
	const int model_h = cascade->getInputSize().size[0];
	const int model_w = cascade->getInputSize().size[1];
	print("Cascade [%s]: width = %d, height = %d\n", model_filename, model_w, model_h);
	CHECK_FATAL(image.resize(model_h, model_w, 1) == true);

	// Load the bindata header
	File file;
	if (file.open(data_filename, "r") == false)
	{
		print("ERROR: loading bindata [%s]!\n", data_filename);
		delete cascade;
		return 1;
	}
	int n_samples;
	int sample_size;
	if (file.read(&n_samples, sizeof(int), 1) != 1)
	{
		print("ERROR: reading <n_samples> from bindata [%s]!\n", data_filename);
		delete cascade;
		return 1;
	}
	if (file.read(&sample_size, sizeof(int), 1) != 1)
	{
		print("ERROR: reading <sample_size> from bindata [%s]!\n", data_filename);
		delete cascade;
		return 1;
	}

	// Check bindata parameters
	print("Bindata: n_samples = %d, sample_size = %d\n", n_samples, sample_size);
	if (n_samples < 1 || sample_size != model_w * model_h)
	{
		print("ERROR: invalid <n_samples> or <sample_size>! from bindata [%s]!\n", data_filename);
		delete cascade;
		return 1;
	}

	// Check each sample if it's a feature or not
	int n_patterns = 0;
	for (int j = 0; j < n_samples; j ++)
	{
		// Load the sample in some image
		for (int y = 0; y < model_h; y ++)
			for (int x = 0; x < model_w; x ++)
			{
				float value;
				if (file.read(&value, sizeof(float), 1) != 1)
				{
					print("ERROR: could not read pixel [y=%d,x=%d] from sample [%d/%d]!\n",
						y, x, j + 1, n_samples);
					delete cascade;
					return 1;
				}

				const short pixel = (short)(value * 255.0f + 0.5f);
				image.set(y, x, 0, pixel);
			}
		print(" ... loaded sample [%d/%d]\r", j + 1, n_samples);

		// Save the loaded image
		if (false)
		{
			char str[1024];
			sprintf(str, "input.%d.jpg", j);

			xtprobeImageFile ifile;
			ifile.save(image, str);
		}

		// Run the cascade
		if (cascade->forward(image) == false)
		{
			print("ERROR: failed to run the cascade on the image [%d/%d]!\n",
				j + 1, n_samples);
			delete cascade;
			return 1;
		}
		//print("CONFIDENCE = %f\n", cascade->getConfidence());
		n_patterns += cascade->isPattern() ? 1 : 0;
	}
	print("\r");

	// Print the results
	print("---------------------------------------------------\n");
	print(">>> detection rate: [%d/%d] = %f%% <<<\n",
		n_patterns, n_samples, 100.0f * (n_patterns + 0.0f) / (n_samples + 0.0f));
	print("---------------------------------------------------\n");

	// Cleanup
	delete cascade;

        print("\nOK\n");

	return 0;
}

