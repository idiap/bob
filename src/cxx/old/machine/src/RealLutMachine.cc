/**
 * @file cxx/old/machine/src/RealLutMachine.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "machine/RealLutMachine.h"
#include "sp/spCore.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////
// Constructor

    RealLutMachine::RealLutMachine() : Machine()
    {
        n_bins = 0;
        lut = NULL;
        min = 0;
        max = 0;
    }
///////////////////////////////////////////////////////////////////////
    bool RealLutMachine::forward(const Tensor& input)
    {
        if (lut == NULL)
        {
            Torch::error("RealLutMachine::forward() no LUT available.");
            return false;
        }

        if (m_core == NULL)
        {
            Torch::error("RealLutMachine::forward() no core available.");
            return false;
        }

        if (m_core->process(input) == false)
        {
            Torch::error("RealLutMachine::forward() core failed.");
            return false;
        }

        DoubleTensor *core_t_output = (DoubleTensor*) &m_core->getOutput(0);

        double feature = core_t_output->get(0);

        double lut_output_;

        if (feature < min) lut_output_ = lut[0];
        else if (feature > max) lut_output_ = lut[n_bins-1];
        else
        {
            int index = (int) floor(n_bins * (feature - min) / (max - min));
            lut_output_ = lut[index];
        }

        m_output.set(0, lut_output_);

        return true;
    }
////////////////////////////////////////////////////////////////////////////////
    bool RealLutMachine::loadFile(File& file)
    {

        verbose = getBOption("verbose");
        int id;
        if (file.taggedRead(&id, 1, "ID") != 1)
        {
            Torch::message("RealLutMachine::load - failed to Read <ID> field!\n");
            return false;
        }

        if (id != getID())
        {
            Torch::message("RealLutMachine::load - invalid <ID>, this is not a RealLutMachine model!\n");
            return false;
        }



        if (file.taggedRead(&n_bins, 1, "N_BINS") != 1)
        {
            Torch::message("RealLutMachine::load - failed to read <n_bins> field!\n");
            return false;
        }

        delete[] lut;
        lut = new double [n_bins];

        if (file.taggedRead(lut, n_bins, "Lut") != n_bins)
        {
            Torch::message("RealLutMachine::load - failed to read <Lut> field!\n");
            return false;
        }

        if (file.taggedRead(&min, 1, "min") != 1)
        {
            Torch::message("RealLutMachine::load - failed to Read <min> field!\n");
            return false;
        }
//
        if (file.taggedRead(&max, 1, "max") != 1)
        {
            Torch::message("RealLutMachine::load - failed to Read <max> field!\n");
            return false;
        }



        int idCore;
        if (file.taggedRead(&idCore, 1, "CoreID") != 1)
        {
            Torch::message("RealLutMachine::load - failed to read <CoreID> field!\n");
            return false;
        }

        if (verbose);
        print("RealLutMachine::LoadFile()\n");

        m_core = spCoreManager::getInstance().get(idCore);
        if (m_core == 0)
        {
            Torch::message("RealLutMachine::load - invalid <CoreID> field!\n");
            return false;
        }

        if (m_core->loadFile(file) == false)
        {
            Torch::message("RealLutMachine::load - the spCore cannot be loaded!\n");
            return false;
        }

        return true;
    }
////////////////////////////////////////////////////////////////////////
    bool RealLutMachine::saveFile(File& file) const
    {



        const int id = getID();
        if (file.taggedWrite(&id, 1, "ID") != 1)
        {
            Torch::message("RealLutMachine::save - failed to write <ID> field!\n");
            return false;
        }
        if (verbose);
        print("ID of the machine : %d\n",id);

        if (file.taggedWrite(&n_bins, 1, "N_BINS") != 1)
        {
            Torch::message("RealLutMachine::save - failed to write <n_bins> field!\n");
            return false;
        }


        // print("size of Lut %d\n",sizeof(lut));
        if (file.taggedWrite(lut, n_bins, "Lut") != n_bins)
        {
            Torch::message("RealLutMachine::save - failed to write <Lut> field!\n");
            return false;
        }

        if (file.taggedWrite(&min, 1, "min") != 1)
        {
            Torch::message("RealLutMachine::save - failed to write <min> field!\n");
            return false;
        }
//
        if (file.taggedWrite(&max, 1, "max") != 1)
        {
            Torch::message("RealLutMachine::save - failed to write <max> field!\n");
            return false;
        }

        if (verbose);
        print("RealLutMachine::saveFile()\n");

        //print("  max = %g\n",max);



        if (m_core == NULL ||  m_core->saveFile(file) == false)
        {
            Torch::message("RealLutMachine::save - cannot save spCore!\n");
            return false;
        }

        //return true;
        return true;
    }
////////////////////////////////////////////////////////////////////////
    void RealLutMachine::setParams(double min_, double max_, int n_bins_, double *lut_)
    {
        verbose = getBOption("verbose");
        if (verbose);
        Torch::print("   RealLutMachine::setParams()\n");

        min = min_;
        max = max_;
        n_bins = n_bins_;
        lut = lut_;
    }
/////////////////////////////////////////////////////////////////
    RealLutMachine::~RealLutMachine()
    {
        delete[] lut;
    }

}

