/**
 * @file cxx/old/machine/src/StumpMachine.cc
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
#include "machine/StumpMachine.h"
#include "sp/spCore.h"

namespace Torch
{

///////////////////////////////////////////////////////////////////////////
// Constructor

    StumpMachine::StumpMachine() : Machine()
    {
        //feature_id = -1;
        threshold = 0.0;
        direction = 0;
        m_output.resize(1);
    }
//////////////////////////////////////////////////////////////////////////
    bool StumpMachine::forward(const Tensor& input)
    {
        if (m_core == NULL)
        {
            Torch::error("StumpMachine::forward() no core available.");
            return false;
        }

        //DoubleTensor* t_input = (DoubleTensor*) input;

        if (m_core->process(input) == false)
        {
            Torch::error("StumpMachine::forward() core failed.");
            return false;
        }

        DoubleTensor *core_t_output = (DoubleTensor*) &m_core->getOutput(0);

        double feature = core_t_output->get(0);

        //  print("....feature......%f\n",feature);
        double stump_output_;

        if (direction == 1)
        {
            if (feature >= threshold) stump_output_ = 1.0;
            else stump_output_ = -1.0;
        }
        else
        {
            if (feature < threshold) stump_output_ = 1.0;
            else stump_output_ = -1.0;
        }

        m_output.set(0, stump_output_);

        return true;
    }
////////////////////////////////////////////////////////////////////////////
    bool StumpMachine::loadFile(File& file)
    {
        int id;

        verbose = getBOption("verbose");
        if (file.taggedRead(&id, 1, "ID") != 1)
        {
            Torch::message("StumpMachine::load - failed to Read <ID> field!\n");
            return false;
        }
        if (id != getID())
        {
            Torch::message("StumpMachine::load - invalid <ID>, this is not a StumpMachine model!\n");
            return false;
        }

        if (file.taggedRead(&threshold, 1, "THRESHOLD") != 1)
        {
            Torch::message("StumpMachine::load - failed to read <threshold> field!\n");
            return false;
        }


        if (file.taggedRead(&direction, 1, "DIRECTION") != 1)
        {
            Torch::message("StumpMachine::load - failed to read <direction> field!\n");
            return false;
        }

        int idCore;
        if (file.taggedRead(&idCore, 1, "CoreID") != 1)
        {
            Torch::message("StumpMachine::load - failed to read <CoreID> field!\n");
            return false;
        }

        if (verbose)
        {
            print("StumpMachine::LoadFile()\n");
            print("   threshold = %g\n", threshold);
            print("   direction = %d\n", direction);
            print("   idCore = %d\n",idCore);
        }

        m_core = spCoreManager::getInstance().get(idCore);
        if (m_core == 0)
        {
            Torch::message("StumpMachine::load - invalid <CoreID> field!\n");
            return false;
        }

        if (m_core->loadFile(file) == false)
        {
            Torch::message("StumpMachine::load - the spCore cannot be loaded!\n");
            return false;
        }

        return true;
    }
///////////////////////////////////////////////////////////////////////////////////////////
    bool StumpMachine::saveFile(File& file) const
    {

        const int id = getID();
        if (file.taggedWrite(&id, 1, "ID") != 1)
        {
            Torch::message("StumpMachine::save - failed to write <ID> field!\n");
            return false;
        }

        if (verbose)
            print("ID of the machine : %d\n",id);
        if (file.taggedWrite(&threshold, 1, "THRESHOLD") != 1)
        {
            Torch::message("StumpMachine::save - failed to write <threshold> field!\n");
            return false;
        }

        if (file.taggedWrite(&direction, 1, "DIRECTION") != 1)
        {
            Torch::message("StumpMachine::save - failed to write <direction> field!\n");
            return false;
        }

        if (verbose)
        {
            print("StumpMachine::saveFile()\n");
            print("   threshold = %g\n", threshold);
            print("   direction = %d\n", direction);
        }
        if (m_core == NULL ||  m_core->saveFile(file) == false)
        {
            Torch::message("StumpMachine::save - the spCore cannot be saved!\n");
            return false;
        }

        return true;
    }
//////////////////////////////////////////////////////////////////////////////////
    void StumpMachine::setParams(int direction_, float threshold_)
    {

        //Torch::print("   StumpMachine::setParams()\n");
        verbose = getBOption("verbose");

        direction = direction_;
        threshold = threshold_;
    }
////////////////////////////////////////////////////////////////////////////////////
    StumpMachine::~StumpMachine()
    {
    }

}

