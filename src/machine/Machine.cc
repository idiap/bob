#include "machine/Machine.h"
#include "sp/spCore.h"

namespace Torch {

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
        if (file.taggedRead(&id, 1, "ID") != 1)
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
                Torch::message("Torch::loadMachineFromFile - failed to load the model file!\n");
                return 0;
        }

        // OK
        file.close();   // Not needed, but to make it clear!
        return machine;
}

//////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////
// Constructor

Machine::Machine()
	: 	Object(),
		m_parameters(new Parameters()),
		m_size(),
		m_core(0),
		m_output(1)
{
}

///////////////////////////////////////////////////////////////////////////
// Destructor

Machine::~Machine()
{
   	delete m_parameters;
}

///////////////////////////////////////////////////////////////////////////
// Set the model size to use

void Machine::setSize(const TensorSize& size)
{
	m_size = size;
}

///////////////////////////////////////////////////////////////////////////
// Set the region to process (for the spCore, if needed)

void Machine::setRegion(const TensorRegion& region)
{
	m_region = region;
	if (m_core != 0)
	{
		m_core->setRegion(region);
	}
}

///////////////////////////////////////////////////////////////////////////
// Set the spCore to use for feature extraction (if needed)

void Machine::setCore(spCore* core)
{
	m_core = core;
}

///////////////////////////////////////////////////////////////////////////
// Destructor

MachineManager::~MachineManager()
{
	deallocate();
}

///////////////////////////////////////////////////////////////////////////
// Register a new <Machine> with a given ID (supposed to be unique)

bool MachineManager::add(Machine* machine, const char* name)
{
	// Check first if the parameters are ok
	if (machine == 0 || machine->getID() < 1)
	{
		delete machine;
		//Torch::message("MachineManager::add - invalid parameters!\n");
		return false;
	}

	// Check if the <id> is taken
	if (find(machine->getID()) >= 0) // the <id> is taken
	{
		delete machine;
		//Torch::message("MachineManager::add - the ID is taken!\n");
		return false;
	}

	// Resize if needed
	if (m_size >= m_capacity)
	{
		resize(32);
	}

	// Add the Machine
	m_machines[m_size] = machine;
	m_names[m_size] = new char[strlen(name) + 1];
	strcpy(m_names[m_size], name);
	m_size ++;
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Get a copy of the <Machine> (empty, no parameters set) for the given ID
// (returns NULL/0 if the <id> is invalid)
// The new Machine is allocated and should be deallocated by the user!

Machine* MachineManager::get(int id) const
{
        const int index = find(id);
        return index < 0 ? 0 : m_machines[index]->getAnInstance();
}

///////////////////////////////////////////////////////////////////////////
// Get the generic name for the given id
// (returns NULL/0 if the <id> is invalid)

const char* MachineManager::getName(int id) const
{
        const int index = find(id);
        return index < 0 ? 0 : m_names[index];
}

///////////////////////////////////////////////////////////////////////////
// Constructor

MachineManager::MachineManager()
	:	m_machines(0), m_names(0),
		m_size(0),
		m_capacity(0)
{
}

///////////////////////////////////////////////////////////////////////////
// Deallocate memory

void MachineManager::deallocate()
{
	for (int i = 0; i < m_size; i ++)
	{
		delete m_machines[i];
		delete[] m_names[i];
	}
	delete[] m_machines;
	delete[] m_names;
	m_machines = 0;
	m_names = 0;

	m_size = 0;
	m_capacity = 0;
}

///////////////////////////////////////////////////////////////////////////
// Resize the IDs to fit the new <increment> (IDs + Machines)

void MachineManager::resize(int increment)
{
	// Allocate new memory
	Machine** new_machines = new Machine*[m_capacity + increment];
	char** new_names = new char*[m_capacity + increment];

	// Copy the old data
	for (int i = 0; i < m_size; i ++)
	{
		new_machines[i] = m_machines[i];
		new_names[i] = m_names[i];
	}

	// Deallocate the old data
	delete[] m_machines;
	delete[] m_names;

	// OK
	m_machines = new_machines;
	m_names = new_names;
	m_capacity += increment;
}

///////////////////////////////////////////////////////////////////////////
// Returns the machine's index with the given ID (or -1, if not found)

int MachineManager::find(int id) const
{
	for (int i = 0; i < m_size; i ++)
		if (m_machines[i]->getID() == id)
		{
			return i;
		}
	return -1;
}

///////////////////////////////////////////////////////////////////////////

}

