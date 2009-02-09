#ifndef _TORCH5SPRO_MACHINE_H_
#define _TORCH5SPRO_MACHINE_H_

#include "Object.h"
#include "Tensor.h"

namespace Torch {

	class Tensor;
	class DoubleTensor;
	class Machine;

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
		// The new Machine is allocated and should be deallocated by the user!
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

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::Machine:
	//      Process some input using a model (loaded from some file).
	//      The output is a DoubleTensor!
	//
	//      NB: The ouput should be allocated and deallocated by each Machine implementation!
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
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const = 0;

		// Set the input size to use
		virtual bool		setInputSize(const TensorSize& inputSize);

		// Get the ID specific to each Machine
		virtual int		getID() const = 0;

		///////////////////////////////////////////////////////////
		// Access functions

		const TensorSize&	getInputSize() const { return m_inputSize; }
		const DoubleTensor&     getOutput() const;

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// Model size (size of the input tensor to process)
		TensorSize		m_inputSize;

		// The result when run on some data
		DoubleTensor*		m_output;
	};
}

#endif
