#include "IntLutMachine.h"
#include "spCore.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////
// Constructor

    IntLutMachine::IntLutMachine() : Machine()
    {
        n_bins = 0;
        lut = NULL;
        m_output.resize(1);

    }
////////////////////////////////////////////////

    bool IntLutMachine::forward(const Tensor& input)
    {
        if (lut == NULL)
        {
            Torch::error("IntLutMachine::forward() no LUT available.");
            return false;
        }

        if (m_core == NULL)
        {
            Torch::error("IntLutMachine::forward() no core available.");
            return false;
        }

        if (m_core->process(input) == false)
        {
            Torch::error("IntLutMachine::forward() core failed.");
            return false;
        }

        IntTensor *core_t_output = (IntTensor*) &m_core->getOutput(0);

        int feature = (*core_t_output)(0);

        m_output.set(0, lut[feature]);

        return true;
    }

    bool IntLutMachine::loadFile(File& file)
    {

        int id;
        if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
        {
            Torch::message("IntLutMachine::load - failed to Read <ID> field!\n");
            return false;
        }

        if (id != getID())
        {
            Torch::message("IntLutMachine::load - invalid <ID>, this is not an IntLutMachine model!\n");
            return false;
        }



        if (file.taggedRead(&n_bins, sizeof(int), 1, "N_BINS") != 1)
        {
            Torch::message("IntLutMachine::load - failed to read <n_bins> field!\n");
            return false;
        }

        delete[] lut;
        lut = new double [n_bins];

        if (file.taggedRead(lut, sizeof(double), n_bins, "Lut") != n_bins)
        {
            Torch::message("IntLutMachine::load - failed to read <Lut> field!\n");
            return false;
        }
//print("  N_Bins = %d\n", n_bins);
//    for(int i=0;i<n_bins;i++)
//        print("lut.......%f\n",lut[i]);


        int idCore;
        if (file.taggedRead(&idCore, sizeof(int), 1, "CoreID") != 1)
        {
            Torch::message("IntLutMachine::load - failed to read <CoreID> field!\n");
            return false;
        }

        //print("IntLutMachine::LoadFile()\n");

        m_core = spCoreManager::getInstance().get(idCore);
        if (m_core == 0)
        {
            Torch::message("IntLutMachine::load - invalid <CoreID> field!\n");
            return false;
        }

        if (m_core->loadFile(file) == false)
        {
            Torch::message("IntLutMachine::load - the spCore cannot be loaded!\n");
            return false;
        }

        return true;

    }

    bool IntLutMachine::saveFile(File& file) const
    {
        const int id = getID();
        if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
        {
            Torch::message("IntLutMachine::save - failed to write <ID> field!\n");
            return false;
        }

        //print("ID of the machine : %d\n",id);

        if (file.taggedWrite(&n_bins, sizeof(int), 1, "N_BINS") != 1)
        {
            Torch::message("IntLutMachine::save - failed to write <n_bins> field!\n");
            return false;
        }


        // print("size of Lut %d\n",sizeof(lut));
        if (file.taggedWrite(lut, sizeof(double), n_bins, "Lut") != n_bins)
        {
            Torch::message("IntLutMachine::save - failed to write <Lut> field!\n");
            return false;
        }

        if (m_core == NULL || m_core->saveFile(file) == false)
        {
            Torch::message("IntLutMachine::save - cannot save spCore!\n");
            return false;
        }

        return true;
    }

    void IntLutMachine::setParams(int n_bins_, double *lut_)
    {
        n_bins = n_bins_;
        delete[] lut;
        lut = new double [n_bins];
        for (int i = 0; i < n_bins; i ++)
        {
                lut[i] = lut_[i];
        }
    }

    IntLutMachine::~IntLutMachine()
    {
        delete[] lut;
    }

}

