#include "sp/spCore.h"
#include "core/general.h"
#include "core/Tensor.h"

/**
 * \addtogroup libsp_api libSP API
 * @{
 *
 *  The libSP API.
 */
namespace Torch {


////////////////////////////////////////////////////////////////////
// Constructor

spCore::spCore()
	:	Object(),
		m_output(0),
		m_n_outputs(0)
{
}

////////////////////////////////////////////////////////////////////
// Destructor

spCore::~spCore()
{
	cleanup();
}

////////////////////////////////////////////////////////////////////
// Deallocate allocated output tensors

void spCore::cleanup()
{
	for (int i = 0; i < m_n_outputs; i ++)
	{
		delete m_output[i];
	}
	delete[] m_output;

	m_output = 0;
	m_n_outputs = 0;
}

////////////////////////////////////////////////////////////////////
// Process some input tensor

bool spCore::process(const Tensor& input)
{
	// Check if the input tensor has the right dimensions and type
	if (checkInput(input) == false)
	{
		Torch::message("Torch::spCore::process - the input tensor is invalid!\n");
		return false;
	}

	// Allocate (if needed) the output tensor given the input tensor dimensions
	if (allocateOutput(input) == false)
	{
		Torch::message("Torch::spCore::process - cannot allocate output tensors!\n");
		return false;
	}

	// OK, now do the processing ...
	return processInput(input);
}

////////////////////////////////////////////////////////////////////
// Access the results

int spCore::getNOutputs() const
{
	return m_n_outputs;
}

const Tensor& spCore::getOutput(int index) const
{
	if (index < 0 || index >= m_n_outputs)
	{
		Torch::error("Torch::spCore::getOutput - invalid index!");
	}
	return *m_output[index];
}

////////////////////////////////////////////////////////////////////
// Change the region of the input tensor to process

void spCore::setRegion(const TensorRegion& region)
{
	m_region = region;
}

////////////////////////////////////////////////////////////////////
// Change the model size (if used with some machine)

void spCore::setModelSize(const TensorSize& modelSize)
{
	m_modelSize = modelSize;
}

////////////////////////////////////////////////////////////////////
/// Loading/Saving the content from files (<em>not the options</em>)

bool spCore::loadFile(File& file)
{
	return true;
}

bool spCore::saveFile(File& file) const
{
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Destructor

spCoreManager::~spCoreManager()
{
	deallocate();
}

///////////////////////////////////////////////////////////////////////////
// Register a new <spCore> with a given ID (supposed to be unique)

bool spCoreManager::add(spCore* core, const char* name)
{
	// Check first if the parameters are ok
	if (core == 0 || core->getID() < 1)
	{
		//Torch::message("spCoreManager::add - invalid parameters!\n");
		return false;
	}

	// Check if the <id> is taken
	if (find(core->getID()) >= 0) // the <id> is taken
	{
		//Torch::message("spCoreManager::add - the ID is taken!\n");
		return false;
	}

	// Resize if needed
	if (m_size >= m_capacity)
	{
		resize(32);
	}

	// Add the spCore
	m_spcores[m_size] = core;
	m_names[m_size] = new char[strlen(name) + 1];
	strcpy(m_names[m_size], name);
	m_size ++;
	return true;
}

///////////////////////////////////////////////////////////////////////////
// Get a copy of the <spCore> (empty, no parameters set) for the given ID
// (returns NULL/0 if the <id> is invalid)
// The new spCore is automatically deallocated!

spCore* spCoreManager::get(int id) const
{
        const int index = find(id);
        return index < 0 ? 0 : m_spcores[index]->getAnInstance();
}

///////////////////////////////////////////////////////////////////////////
// Get the generic name for the given id
// (returns NULL/0 if the <id> is invalid)

const char* spCoreManager::getName(int id) const
{
        const int index = find(id);
        return index < 0 ? 0 : m_names[index];
}

///////////////////////////////////////////////////////////////////////////
// Constructor

spCoreManager::spCoreManager()
	:	m_spcores(0), m_names(0),
		m_size(0),
		m_capacity(0)
{
}

///////////////////////////////////////////////////////////////////////////
// Deallocate memory

void spCoreManager::deallocate()
{
	for (int i = 0; i < m_size; i ++)
	{
		delete[] m_names[i];
	}
	delete[] m_spcores;
	delete[] m_names;
	m_spcores = 0;
	m_names = 0;

	m_size = 0;
	m_capacity = 0;
}

///////////////////////////////////////////////////////////////////////////
// Resize the IDs to fit the new <increment> (IDs + Machines)

void spCoreManager::resize(int increment)
{
	// Allocate new memory
	spCore** new_spcores = new spCore*[m_capacity + increment];
	char** new_names = new char*[m_capacity + increment];

	// Copy the old data
	for (int i = 0; i < m_size; i ++)
	{
		new_spcores[i] = m_spcores[i];
		new_names[i] = m_names[i];
	}

	// Deallocate the old data
	delete[] m_spcores;
	delete[] m_names;

	// OK
	m_spcores = new_spcores;
	m_names = new_names;
	m_capacity += increment;
}

///////////////////////////////////////////////////////////////////////////
// Returns the machine's index with the given ID (or -1, if not found)

int spCoreManager::find(int id) const
{
	for (int i = 0; i < m_size; i ++)
		if (m_spcores[i]->getID() == id)
		{
			return i;
		}
	return -1;
}

///////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////

}


/**
 * @}
 */

