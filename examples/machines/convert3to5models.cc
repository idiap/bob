#include "File.h"
#include "Machines.h"
#include "spCores.h"

using namespace Torch;

int mct_to_lbp[512];

///////////////////////////////////////////////////////////////////////////
// Initialize the MCT to LBP code transformation table
///////////////////////////////////////////////////////////////////////////

void initMCT_to_LBP()
{
        static const int conv_powers[9] =
                {
                        // LBP -> MCT (power index)
                        0,
                        1,
                        2,
                        7,
                        8,
                        3,
                        6,
                        5,
                        4
                };

        int sum_lbp = 0;
        for (unsigned mct = 0; mct < 512; mct ++)
        {
                unsigned int lbp = 0;

                for (int j = 0; j < 9; j ++)
                {
                        const int power = (0x01 << (8 - j));
                        const int mct_pw = (mct & (power)) >> (8 - j);
                        //print("%d, ", mct_pw);
                        lbp += mct_pw * (0x01 << (8 - conv_powers[j]));
                }
                mct_to_lbp[mct] = lbp;

                sum_lbp += lbp;
                //print(": mct = %d -> lbp = %d\n", mct, lbp);
        }

        //print("sum_lbp = %d, should be = %d\n", sum_lbp, 512 * 511/ 2);
        CHECK_FATAL(sum_lbp == 512 * 511/ 2);
}

///////////////////////////////////////////////////////////////////////////
// Convert the Torch3 cascade model file to the torch5spro model
///////////////////////////////////////////////////////////////////////////

bool convert(File& file_in, File& file_out)
{
        CascadeMachine cascade_machine;

        // Get the model size
	int model_w, model_h;
	if (file_in.taggedRead(&model_w, sizeof(int), 1, "WIDTH") != 1)
	{
	        return false;
	}
	if (file_in.taggedRead(&model_h, sizeof(int), 1, "HEIGHT") != 1)
        {
        	return false;
	}

	TensorSize modelsize(model_h, model_w);
	TensorRegion tregion(0, 0, model_h, model_w);

	// Create the machine stages
	int n_stages;
        if (file_in.taggedRead(&n_stages, sizeof(int), 1, "N_STAGES") != 1)
        {
        	return false;
        }
        if (cascade_machine.resize(n_stages) == false)
        {
        	return false;
        }
        print("N_STAGES = %d\n", n_stages);

        // For each stage ... read the thresholds and the number of machines
        for (int s = 0; s < n_stages; s ++)
        {
                // Threshold
		float threshold;
		if (file_in.taggedRead(&threshold, sizeof(float), 1, "THRESHOLD") != 1)
		{
			return false;
		}
		if (cascade_machine.setThreshold(s, threshold) == false)
		{
			return false;
		}
		print("\t[%d/%d] THRESHOLD = %f\n", s + 1, n_stages, threshold);

		// Number of machines per stage
		int n_trainers;
		if (file_in.taggedRead(&n_trainers, sizeof(int), 1, "N_TRAINERS") != 1)
		{
			return false;
		}
		if (cascade_machine.resize(s, n_trainers) == false)
		{
			return false;
		}
		print("\t[%d/%d] N_TRAINERS = %d\n", s + 1, n_stages, n_trainers);
        }

        // For each stage ... load the machines
	for (int s = 0; s < n_stages; s ++)
	{
                const int n_trainers = cascade_machine.getNoMachines(s);
                float* weights = new float[n_trainers];

		// Read the weights
		if (file_in.taggedRead(weights, sizeof(float), n_trainers, "WEIGHTS") != n_trainers)
		{
		        delete[] weights;
			return false;
		}
		for (int n = 0; n < n_trainers; n ++)
		{
                        cascade_machine.setWeight(s, n, weights[n]);
		}
		delete[] weights;
		//print("\t[%d/%d] %d weights loaded\n", s + 1, n_stages, n_trainers);

		// Load the machines
		for (int n = 0; n < n_trainers; n ++)
		{
                        IntLutMachine* lbp_machine = new IntLutMachine;
                        ipLBP8R* iplbp = new ipLBP8R;
			if (	iplbp->setBOption("ToAverage", true) == false ||
				iplbp->setBOption("AddAvgBit", true) == false ||
				iplbp->setBOption("Uniform", false) == false ||
				iplbp->setBOption("RotInvariant", false) == false)
			{
				return false;
			}

			iplbp->setModelSize(modelsize);
			iplbp->setRegion(tregion);

                        // Read XY position
                        int pos_xy;
                        if (file_in.taggedRead(&pos_xy, sizeof(int), 1, "LOCATION") != 1)
                        {
                                delete lbp_machine;
                                return false;
                        }
                        const int pos_x = pos_xy % (model_w - 2) + 1;
                        const int pos_y = pos_xy / (model_w - 2) + 1;
                        if (pos_x < 1 || pos_y < 1 || pos_x > 17 || pos_y > 17)
                        {
                              print("\t\t[%d/%d]: pos_xy = %d => pos_x = %d, pos_y = %d\n",
                                      n + 1, n_trainers, pos_xy, pos_x, pos_y);
                        }

                        // Read LUT
                        const int n_lbp_kernels = 512;
                        float* lut = new float[n_lbp_kernels];
                        if (file_in.taggedRead(lut, sizeof(float), n_lbp_kernels, "LUT") != n_lbp_kernels)
                        {
                                delete[] lut;
                                delete lbp_machine;
                                return false;
                        }

                        // Transform the old MCT code to the new one (the bit order!)
                        // [0 1 2]              [0 1 2]
                        // [3 4 5]      =>      [7 8 3]
                        // [6 7 8]              [6 5 4]
                        double* lut_double = new double[n_lbp_kernels];
                        for (int i = 0; i < n_lbp_kernels; i ++)
                        {
                                lut_double[mct_to_lbp[i]] = (double)lut[i];
                        }
                        delete[] lut;

                        // Configure the LBP machine
			iplbp->setXY(pos_x, pos_y);
			lbp_machine->setParams(n_lbp_kernels, lut_double);
			lbp_machine->setCore(iplbp);

                        // Add the LBP machine to the cascade
                        if (cascade_machine.setMachine(s, n, lbp_machine) == false)
                        {
                                delete lbp_machine;
                                return false;
                        }
		}
	}

	// Force the model size to all Machines
	cascade_machine.setSize(modelsize);

	// OK, just let the CascadeMachine object to write his structure to the output file
        return cascade_machine.saveFile(file_out);
}

///////////////////////////////////////////////////////////////////////////
// Main
///////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
	if (argc != 3)
	{
		print("Parameters: <torch3 model filename> <torch5spro model filename>!\n");
		return -1;
	}

        File file_in;
        File file_out;

        // Initialize the MCT to LBP code transformation table
        initMCT_to_LBP();

	const char* in_filename = argv[1];
	const char* out_filename = argv[2];

	// Open the input/output file
	CHECK_FATAL(file_in.open(in_filename, "r") == true);
	CHECK_FATAL(file_out.open(out_filename, "w+") == true);

	// Debug
	print("---------------------------------------------------\n");
	print("INPUT : [%s] successfully opened!\n", in_filename);
	print("OUTPUT: [%s] successfully opened!\n", out_filename);
	print("---------------------------------------------------\n");

	// Do the conversion
	if (convert(file_in, file_out) == false)
	{
		print("Failed to convert!\n\n");
	}
	else
	{
		// Cleanup
		file_in.close();
		file_out.close();

		print("Conversion finished!\n\n");

		// Test the loading
		{
			CascadeMachine cascade;
			CHECK_FATAL(file_in.open(out_filename, "r") == true);
			if (cascade.loadFile(file_in) == true)
			{
				print(">>>>>>>>>>>>>> CHECKED! <<<<<<<<<<<<<<<\n\n");
			}
			else
			{
				print(">>>>>>>>>>>>>> The converted file model is NOT valid! <<<<<<<<<<<<<<\n\n");
			}
			file_in.close();
		}
	}
	print("---------------------------------------------------\n");

        print("\nOK\n");

	return 0;
}
