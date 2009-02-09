#include "Machine.h"
#include "Tensor.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

Machine::Machine()
	: 	Object(),
		m_inputSize(),
		m_output(0)
{
}

///////////////////////////////////////////////////////////////////////////
// Destructor

Machine::~Machine()
{
}

///////////////////////////////////////////////////////////////////////////
// Set the input size to use

bool Machine::setInputSize(const TensorSize& inputSize)
{
	m_inputSize = inputSize;
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Get the model's output

const DoubleTensor& Machine::getOutput() const
{
        if (m_output == 0)
        {
                Torch::error("Machine::getOutput - invalid DoubleTensor!\n");
        }

        return *m_output;
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

