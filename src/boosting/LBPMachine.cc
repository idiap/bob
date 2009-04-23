#include "LBPMachine.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

LBPMachine::LBPMachine() : Machine()
{
   	n_bins = 0;
	lut = NULL;
	m_output = new DoubleTensor(1);

}

bool LBPMachine::forward(const Tensor& input)
{
   	if(lut == NULL)
	{
	   	Torch::error("LBPMachine::forward() no LUT available.");
		return false;
	}

   	if(m_core == NULL)
	{
	   	Torch::error("LBPMachine::forward() no core available.");
		return false;
	}

	if(m_core->process(input) == false)
	{
	   	Torch::error("LBPMachine::forward() core failed.");
		return false;
	}

	IntTensor *core_t_output = (IntTensor*) &m_core->getOutput(0);

	int feature = core_t_output->get(0);

  //  print("feature %d\n",feature);
	double lut_output_;


        //{
		//int index = (int) floor(n_bins * (feature - min) / (max - min));

		lut_output_ = lut[feature];
	//}

	DoubleTensor* t_output = (DoubleTensor*) m_output;
	(*t_output)(0) = lut_output_;

	return true;
}

bool LBPMachine::loadFile(File& file)
{

    int id;
    if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("LBPMachine::load - failed to Read <ID> field!\n");
		return false;
	}


    if (file.taggedRead(&n_bins, sizeof(int), 1, "N_BINS") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <n_bins> field!\n");
		return false;
	}

    lut = new double [n_bins];

     if (file.taggedRead(lut, sizeof(double), n_bins, "Lut") != n_bins)
	{
		Torch::message("LBPMachine::load - failed to read <Lut> field!\n");
		return false;
	}
print("  N_Bins = %d\n", n_bins);
//    for(int i=0;i<n_bins;i++)
//        print("lut.......%f\n",lut[i]);


    int idCore;
    if (file.taggedRead(&idCore, sizeof(int), 1, "CoreID") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <CoreID> field!\n");
		return false;
	}

    print("LBPMachine::LoadFile()\n");

	print("   idCore = %d\n",idCore);
	spCoreManager* spC = new spCoreManager();
	m_core = spC->getCore(idCore);
	m_core->loadFile(file);
	return true;

}

bool LBPMachine::saveFile(File& file) const
{


    const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <ID> field!\n");
		return false;
	}
	print("ID of the machine : %d\n",id);

	if (file.taggedWrite(&n_bins, sizeof(int), 1, "N_BINS") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <n_bins> field!\n");
		return false;
	}


    print("size of Lut %d\n",sizeof(lut));
    if (file.taggedWrite(lut, sizeof(double), n_bins, "Lut") != n_bins)
	{
		Torch::message("LBPMachine::save - failed to write <Lut> field!\n");
		return false;
	}

//	 if (file.taggedWrite(&min, sizeof(double), 1, "min") != 1)
//	{
//		Torch::message("LutMachine::save - failed to write <min> field!\n");
//		return false;
//	}
//
//	 if (file.taggedWrite(&max, sizeof(double), 1, "max") != 1)
//	{
//		Torch::message("LBPMachine::save - failed to write <max> field!\n");
//		return false;
//	}


   	print("LBPMachine::saveFile()\n");

	//print("  max = %g\n",max);



	m_core->saveFile(file);

	return true;
}

void LBPMachine::setParams(int n_bins_, double *lut_)
{
	Torch::print("   LBPMachine::setParams()\n");

	//min = min_;
	//max = max_;
	n_bins = n_bins_;
	lut = lut_;
}

LBPMachine::~LBPMachine()
{
}

}

