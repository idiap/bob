#include "Machines.h"
#include "File.h"

namespace Torch
{

//////////////////////////////////////////////////////////////////////////
// Load a generic machine from some file
// Returns <0/NULL> if some error,
//      otherwise you are responsible for deallocating the Machine

Machine* loadMachineFromFile(const char* filename)
{
        // Just open the file to get the Machine's ID
        File file;
        if (file.open(filename, "r") == false)
        {
                Torch::message("Torch::loadMachineFromFile - cannot open file!\n");
                return 0;
        }
        int id;
        if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
        {
                Torch::message("Torch::loadMachineFromFile - cannot read the <ID> tag!\n");
                return 0;
        }

        // Get the machine for this id
        Machine* machine = MachineManager::getInstance().get(id);
        if (machine == 0)
        {
                Torch::message("Torch::loadMachineFromFile - invalid <ID>!\n");
                return 0;
        }

        // Use this machine to load the file
        file.rewind();
        if (machine->loadFile(file) == false)
        {
                delete machine;
                Torch::message("Torch::loadMachineFromFile - failed to load the model file!\n");
                return 0;
        }

        // OK
        file.close();   // Not needed, but to make it clear!
        return machine;
}

//////////////////////////////////////////////////////////////////////////
}
