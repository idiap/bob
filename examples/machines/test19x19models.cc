#include "File.h"
#include "Image.h"
#include "xtprobeImageFile.h"
#include "Machines.h"
#include <cCHECK_FATAL>

using namespace Torch;

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
        // Check arguments
        if (argc != 2)
        {
                print("\nUsage: <test19x19models> <bindata file to test>\n\n");
                return 1;
        }
        const char* data_filename = argv[1];

        static const int n_models = 4;
        static const char* model_filenames[n_models] =
                {
                        "../data/models/facedetection/frontal/mct4.cascade",
                        "../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade",
                        "../data/models/facedetection/frontal/mct4-2-5-10-50.cascade",
                        "../data/models/facedetection/frontal/mct4-2-5-10-50-200.cascade"
                };

        Image image;

        // Test each model
        for (int i = 0; i < n_models; i ++)
        {
                // Load the cascade machine
                CascadeMachine* cascade = (CascadeMachine*)Torch::loadMachineFromFile(model_filenames[i]);
                if (cascade == 0)
                {
                        print("ERROR: loading model [%s]!\n", model_filenames[i]);
                        continue;
                }
                print("Cascade [%s]: width = %d, height = %d\n",
                        model_filenames[i], cascade->getModelWidth(), cascade->getModelHeight());
                CHECK_FATAL(image.resize(cascade->getModelWidth(), cascade->getModelHeight(), 1) == true);

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
                if (n_samples < 1 || sample_size != cascade->getModelWidth() * cascade->getModelHeight())
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
                        for (int y = 0; y < cascade->getModelHeight(); y ++)
                                for (int x = 0; x < cascade->getModelWidth(); x ++)
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
                                ifile.open(str, "w");
                                image.saveImage(ifile);
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
        }

        print("\nOK\n");

	return 0;
}

