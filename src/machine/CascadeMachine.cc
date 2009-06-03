#include "CascadeMachine.h"
#include "Tensor.h"
#include "File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Reset the number of machines

bool CascadeMachine::Stage::resize(int n_machines)
{
	if (n_machines < 1)
	{
		return false;
	}

	// OK, allocate the new machines
	if (n_machines == m_n_machines)
	{
	}
	else
	{
		deallocate();
		m_n_machines = n_machines;
		m_machines = new Machine*[m_n_machines];
		m_weights = new double[m_n_machines];
		for (int i = 0; i < m_n_machines; i ++)
		{
			m_machines[i] = 0;
			m_weights[i] = 0.0;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Deallocate memory

void CascadeMachine::Stage::deallocate()
{
	delete[] m_machines;
	delete[] m_weights;
	m_machines = 0;
	m_weights = 0;
	m_n_machines = 0;
}

//////////////////////////////////////////////////////////////////////////
// Set a new machine

bool CascadeMachine::Stage::setMachine(int i_machine, Machine* machine)
{
	if (i_machine < 0 || i_machine >= m_n_machines || machine == 0)
	{
		return false;
	}

	// OK
	m_machines[i_machine] = machine;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Set a new weight for some machine

bool CascadeMachine::Stage::setWeight(int i_machine, double weight)
{
	if (i_machine < 0 || i_machine >= m_n_machines)
	{
		Torch::error("CascadeMachine::Stage::setWeight - invalid parameters!\n");
		return false;
	}

	// OK
	m_weights[i_machine] = weight;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Set the model size to use (need to set the model size to each <Machine>) - overriden

void CascadeMachine::setSize(const TensorSize& size)
{
	Machine::setSize(size);

	// OK, set the model size to each <Machine>
	for (int s = 0; s < m_n_stages; s ++)
	{
		Stage& stage = m_stages[s];
		for (int n = 0; n < stage.m_n_machines; n ++)
		{
			Machine* machine = stage.m_machines[n];
			if (machine != 0)
			{
				machine->setSize(size);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// Access functions

double CascadeMachine::Stage::getWeight(int i_machine) const
{
	if (i_machine < 0 || i_machine >= m_n_machines)
	{
		Torch::error("CascadeMachine::Stage::getWeight - invalid parameters!\n");
	}

	return m_weights[i_machine];
}

Machine* CascadeMachine::Stage::getMachine(int i_machine)
{
	if (i_machine < 0 || i_machine >= m_n_machines)
	{
		Torch::error("CascadeMachine::Stage::getMachine - invalid parameters!\n");
	}

	return m_machines[i_machine];
}

const Machine* CascadeMachine::Stage::getMachine(int i_machine) const
{
	if (i_machine < 0 || i_machine >= m_n_machines)
	{
		Torch::error("CascadeMachine::Stage::getMachine - invalid parameters!\n");
	}

	return m_machines[i_machine];
}

//////////////////////////////////////////////////////////////////////////
// Constructor

CascadeMachine::CascadeMachine()
	:	Classifier(),
		m_stages(0), m_n_stages(0),
                m_fast_output(0)
{
        // Allocate the output
	m_output.resize(1);
	const DoubleTensor* t_output = (DoubleTensor*)&m_output;
	m_fast_output = t_output->t->storage->data + t_output->t->storageOffset;
}

//////////////////////////////////////////////////////////////////////////
// Destructor

CascadeMachine::~CascadeMachine()
{
	deallocate();
}

//////////////////////////////////////////////////////////////////////////
// Deallocate the memory

void CascadeMachine::deallocate()
{
	delete[] m_stages;
	m_stages = 0;
	m_n_stages = 0;
}

//////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool CascadeMachine::forward(const Tensor& input)
{
	m_isPattern = true;
	m_patternClass = 0;

	// Check if the input data passes the cascade stages
	// ... for each stage
	for (int s = 0; s < m_n_stages; s ++)
	{
		const Stage& stage = m_stages[s];

		// ... for each Machine in the stage
		double output = 0.0;
		for (int n = 0; n < stage.m_n_machines; n ++)
		{
			Machine* machine = stage.m_machines[n];
			if (machine == 0)
			{
				Torch::message("CascadeMachine::forward - invalid machine!\n");
				return false;
			}
			machine->setRegion(m_region);
			if (machine->forward(input) == false)
			{
				Torch::message("CascadeMachine::forward - error running the machine!\n");
				return false;
			}
			output += machine->getOutput().get(0) * stage.m_weights[n];
		}

		// Check if rejected
		*m_fast_output = output;
		if (output < stage.m_threshold)
		{
		        m_isPattern = false;
			break;
		}
	}

	// OK
	m_confidence = *m_fast_output;
	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool CascadeMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("CascadeMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("CascadeMachine::load - invalid <ID>, this is not a CascadeMachine model!\n");
		return false;
	}

	// Get the model size
	TensorSize inputSize;
	if (file.taggedRead(&inputSize, sizeof(TensorSize), 1, "INPUT_SIZE") != 1)
	{
		Torch::message("CascadeMachine::load - failed to read <INPUT_SIZE> field!\n");
		return false;
	}

	// Create the machine stages
	int n_stages;
        if (file.taggedRead(&n_stages, sizeof(int), 1, "N_STAGES") != 1)
        {
        	Torch::message("CascadeMachine::load - failed to read <N_STAGES> field!\n");
        	return false;
        }
        if (resize(n_stages) == false)
        {
        	return false;
        }

	// For each stage ...
	for (int s = 0; s < n_stages; s ++)
	{
		// Threshold
		double threshold;
		if (file.taggedRead(&threshold, sizeof(double), 1, "THRESHOLD") != 1)
		{
			Torch::message("CascadeMachine::load - failed to read <THRESHOLD> field!\n");
			return false;
		}
		if (setThreshold(s, threshold) == false)
		{
			return false;
		}

		// Number of machines per stage
		int n_trainers;
		if (file.taggedRead(&n_trainers, sizeof(int), 1, "N_TRAINERS") != 1)
		{
			Torch::message("CascadeMachine::load - failed to read <N_TRAINERS> field!\n");
		}
		if (resize(s, n_trainers) == false)
		{
			return false;
		}

		// Load each machine
		for (int n = 0; n < n_trainers; n ++)
		{
		        // Get the Machine's ID
			int id;
			if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
			{
				Torch::message("CascadeMachine::load - failed to read the Machine's <ID> field!\n");
				return false;
			}
			Machine* machine = MachineManager::getInstance().get(id);
			if (machine == 0)
			{
				Torch::message("CascadeMachine::load - invalid machine ID!\n");
				return false;
			}

			// Load the machine's parameters
			if (machine->loadFile(file) == false)
			{
			        Torch::message("CascadeMachine::load - the [%d/%d]-[%d/%d] machine could not be loaded!\n",
                                        s + 1, n_stages, n + 1, n_trainers);
			        return false;
			}
			if (setMachine(s, n, machine) == false)
			{
				return false;
			}

			// Load the weight of this machine
			double weight;
			if (file.taggedRead(&weight, sizeof(double), 1, "WEIGHT") != 1)
			{
				Torch::message("CascadeMachine::load - failed to read <WEIGHT> field!\n");
				return false;
			}
			if (setWeight(s, n, weight) == false)
			{
				return false;
			}
		}
	}

	// OK, force the model size to all Machines
	setSize(inputSize);
	return true;
}

bool CascadeMachine::saveFile(File& file) const
{
	// Write the ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("CascadeMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the model size
	if (file.taggedWrite(&m_size, sizeof(TensorSize), 1, "INPUT_SIZE") != 1)
	{
		Torch::message("CascadeMachine::save - failed to write <INPUT_SIZE> field!\n");
		return false;
	}

	// Write the number of stages
	if (file.taggedWrite(&m_n_stages, sizeof(int), 1, "N_STAGES") != 1)
        {
        	Torch::message("CascadeMachine::save - failed to write <N_STAGES> field!\n");
        	return false;
        }

	// For each stage ...
	for (int s = 0; s < m_n_stages; s ++)
	{
		const Stage& stage = m_stages[s];

		// Threshold
		if (file.taggedWrite(&stage.m_threshold, sizeof(double), 1, "THRESHOLD") != 1)
		{
			Torch::message("CascadeMachine::save - failed to write <THRESHOLD> field!\n");
			return false;
		}

		// Number of machines per stage
		if (file.taggedWrite(&stage.m_n_machines, sizeof(int), 1, "N_TRAINERS") != 1)
		{
			Torch::message("CascadeMachine::save - failed to write <N_TRAINERS> field!\n");
		}

		// Save each machine
		for (int n = 0; n < stage.m_n_machines; n ++)
		{
			const Machine* machine = stage.m_machines[n];

			// Write the Machine's ID
			int id = machine->getID();
			if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
			{
				Torch::message("CascadeMachine::save - failed to write the Machine's <ID> field!\n");
				return false;
			}

			// Save the machine's parameters
			if (machine->saveFile(file) == false)
			{
			        Torch::message("CascadeMachine::load - the [%d/%d]-[%d/%d] machine could not be saved!\n",
                                        s + 1, m_n_stages, n + 1, stage.m_n_machines);
				return false;
			}

			// Save the weight of this machine
			if (file.taggedWrite(&stage.m_weights[n], sizeof(double), 1, "WEIGHT") != 1)
			{
				Torch::message("CascadeMachine::save - failed to write <WEIGHT> field!\n");
				return false;
			}
		}
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Machine manipulation (add/remove/set Machines or stages)
// NB: The Cascade will take care of deallocating the SET machines!

bool CascadeMachine::resize(int n_stages)
{
	if (n_stages < 1)
	{
		Torch::message("CascadeMachine::resize - invalid parameters!\n");
		return false;
	}

	deallocate();

	// OK
	m_n_stages = n_stages;
	m_stages = new Stage[m_n_stages];
	return true;
}

bool CascadeMachine::resize(int i_stage, int n_machines)
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::message("CascadeMachine::resize(,) - invalid parameters!\n");
		return false;
	}

	// OK
	Stage& stage = m_stages[i_stage];
	return stage.resize(n_machines);
}

bool CascadeMachine::setMachine(int i_stage, int i_machine, Machine* machine)
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::message("CascadeMachine::setMachine - invalid parameters!\n");
		return false;
	}

	// OK
	Stage& stage = m_stages[i_stage];
	if (stage.setMachine(i_machine, machine))
	{
		// Don't forget to set the model size to this machine too
		machine->setSize(m_size);
		return true;
	}
	return false;
}

bool CascadeMachine::setWeight(int i_stage, int i_machine, double weight)
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::message("CascadeMachine::setWeight - invalid parameters!\n");
		return false;
	}

	// OK
	Stage& stage = m_stages[i_stage];
	return stage.setWeight(i_machine, weight);
}

bool CascadeMachine::setThreshold(int i_stage, double threshold)
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::message("CascadeMachine::setThreshold - invalid parameters!\n");
		return false;
	}

	// OK
	m_stages[i_stage].m_threshold = threshold;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Access functions

int CascadeMachine::getNoMachines(int i_stage) const
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::error("CascadeMachine::getNoMachines - invalid parameters!\n");
	}

	return m_stages[i_stage].m_n_machines;
}

Machine* CascadeMachine::getMachine(int i_stage, int i_machine)
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::error("CascadeMachine::getMachine - invalid parameters!\n");
	}

	return m_stages[i_stage].getMachine(i_machine);
}

const Machine* CascadeMachine::getMachine(int i_stage, int i_machine) const
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::error("CascadeMachine::getMachine - invalid parameters!\n");
	}

	return m_stages[i_stage].getMachine(i_machine);
}

double CascadeMachine::getWeight(int i_stage, int i_machine) const
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::error("CascadeMachine::getWeight - invalid parameters!\n");
	}

	return m_stages[i_stage].getWeight(i_machine);
}

double CascadeMachine::getThreshold(int i_stage) const
{
	if (i_stage < 0 || i_stage >= m_n_stages)
	{
		Torch::error("CascadeMachine::getThreshold - invalid parameters!\n");
	}

	return m_stages[i_stage].m_threshold;
}

//////////////////////////////////////////////////////////////////////////
}
