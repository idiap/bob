#ifndef _TORCH5SPRO_MACHINE_H_
#define _TORCH5SPRO_MACHINE_H_

#include "core/Object.h"
#include "core/Parameters.h"
#include "core/Tensor.h"

/**
 * \addtogroup libmachine_api libMachine API
 * @{
 *
 *  The libMachine API.
 */
namespace Torch {

	class Machine;
	class File;
	class spCore;

	///////////////////////////////////////////////////////////////////////////

        // Load a generic machine from some file
        // Returns <0/NULL> if some error,
        //      otherwise you are responsible for deallocating the Machine
        Machine*                loadMachineFromFile(const char* filename);

        ///////////////////////////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::Machine:
	//      Process some input using a model (loaded from some file).
	//	May use some spCore object to extract features. It doesn't deallocate the given spCore.
	//      The output is a DoubleTensor!
	//
	//	EACH MACHINE SHOULD REGISTER
	//		==> MachineManager::GetInstance().add(new XXXMachine) <==
	//	TO THE MACHINEMANAGER CLASS!!!
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Machine : public Object
	{
	public:

		/// Constructor
		Machine();

		/// Destructor
		virtual ~Machine();

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input) = 0;

		/// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine*	getAnInstance() const = 0;

		// Set the model size to use
		virtual void		setSize(const TensorSize& size);

		// Set the region to process (for the spCore, if needed)
		virtual void		setRegion(const TensorRegion& region);

		// Set the spCore to use for feature extraction (if needed)
		virtual void		setCore(spCore* core = 0);

		// Get the ID specific to each Machine
		virtual int		getID() const = 0;

		// Loading/Saving the content from files (<em>not the options</em>)
		virtual bool		loadFile(File& file) = 0;
		virtual bool		saveFile(File& file) const = 0;

		///////////////////////////////////////////////////////////
		// Access functions

		const TensorSize&	getSize() const { return m_size; }
		const DoubleTensor&     getOutput() const { return m_output; }
		spCore*			getCore() { return m_core; }

		///////////////////////////////////////////////////////////

		// Parameters of the machine
		Parameters*		m_parameters;

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// Model size
		TensorSize		m_size;

		// \c spCore to extract some features (if needed)
		spCore*			m_core;
		TensorRegion		m_region;	// Region where to process

		// The result when run on some data
		DoubleTensor		m_output;
	};

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::MachineManager:
	//	Keeps track of the IDs of each Machine.
	//	Given some ID (e.g. found in a model file), it's possible to retrieve a
	//		machine associated with this ID.
	//
	//	NB: the <Machine>s are deallocated from the MachineManager.
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class MachineManager
	{
	public:

		// Destructor
		~MachineManager();

		// Access to the single instance of this object
		static MachineManager&	getInstance()
		{
			static MachineManager manager;
			return manager;
		}

		// Register a new <Machine> with a given ID (supposed to be unique)
		bool			add(Machine* machine, const char* name);

		// Get a copy of the <Machine> (empty, no parameters set) for the given ID
		// (returns NULL/0 if the <id> is invalid)
		// The new Machine will be automatic deallocated!
		Machine*		get(int id) const;

		// Get the generic name for the given id
		// (returns NULL/0 if the <id> is invalid)
		const char*             getName(int id) const;

		// Get the number of registered machines
		int                     getNoMachines() const { return m_size; }

	private:

		// Constructor
		MachineManager();

		// Deallocate memory
		void			deallocate();

		// Resize the IDs to fit the new <increment>
		void			resize(int increment);

		// Returns the machine's index with the given ID (or -1, if not found)
		int     		find(int id) const;

		///////////////////////////////////////////////////////////////
		// Attributes

		// Machine prototypes for each ID
		Machine**		m_machines;
		char**                  m_names;        // Generic name for each machine
		int			m_size;		// Number of IDs actually used
		int			m_capacity;	// Number of IDs allocated
	};
}

/**
 * @}
 */


/**
@page libMachine Machine: a Machine module

@section intro Introduction

Machine is the Torch module used to process some data. A Machine may or may not be trained by a Trainer.

@section api Documentation
- @ref libmachine_api "libMachine API"

*/

#endif
