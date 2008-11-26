#include "Machine.h"
#include "Tensor.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////
// Constructor

Machine::Machine()
	: 	Object(),
		m_model_w(0), m_model_h(0),
		m_output(0.0)
{
}

///////////////////////////////////////////////////////////////////////////
// Destructor

Machine::~Machine()
{
}

///////////////////////////////////////////////////////////////////////////
// Set the model size to use

bool Machine::setModelSize(int model_w, int model_h)
{
	if (model_w < 1 || model_h < 1)
	{
		return false;
	}

	// OK
	m_model_w = model_w;
	m_model_h = model_h;
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Destructor

MachineManager::~MachineManager()
{
	deallocate();
}

///////////////////////////////////////////////////////////////////////////
// Register a new <Machine> with a given ID (supposed to be unique)

bool MachineManager::add(Machine* machine)
{
        // Check first if the parameters are ok
	if (machine == 0 || machine->getID() < 1)
	{
	        delete machine;
		//Torch::message("MachineManager::add - invalid parameters!\n");
		return false;
	}

	// Check if the <id> is taken
	if (find(machine->getID()) != 0) // the <id> is taken
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
	m_size ++;
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Get a copy of the <Machine> (empty, no parameters set) for the given ID
// (returns NULL/0 if the <id> is invalid)
// The new Machine is allocated and should be deallocated by the user!

Machine* MachineManager::get(int id)
{
	const Machine* proto = find(id);
	return proto == 0 ? 0 : proto->getAnInstance();
}

///////////////////////////////////////////////////////////////////////////
// Constructor

MachineManager::MachineManager()
	:	m_machines(0),
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
	}
	delete[] m_machines;
	m_machines = 0;

	m_size = 0;
	m_capacity = 0;
}

///////////////////////////////////////////////////////////////////////////
// Resize the IDs to fit the new <increment> (IDs + Machines)

void MachineManager::resize(int increment)
{
	// Allocate new memory
	Machine** new_machines = new Machine*[m_capacity + increment];

	// Copy the old data
	for (int i = 0; i < m_size; i ++)
	{
		new_machines[i] = m_machines[i];
	}

	// Deallocate the old data
	delete[] m_machines;

	// OK
	m_machines = new_machines;
	m_capacity += increment;
}

///////////////////////////////////////////////////////////////////////////
// Returns NULL or the pointer to the Machine with the given ID

const Machine* MachineManager::find(int id) const
{
	for (int i = 0; i < m_size; i ++)
		if (m_machines[i]->getID() == id)
		{
			return m_machines[i];
		}
	return 0;
}

///////////////////////////////////////////////////////////////////////////

}

