#ifndef SPCORE_INC
#define SPCORE_INC

#include "Object.h"
#include "Tensor.h"
#include "File.h"

namespace Torch
{
	class Tensor;

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::spCore:
	//      //
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class spCore : public Object
	{
	public:
		/// Constructor
		spCore();

		/// Destructor
		virtual ~spCore();

		/// Loading/Saving the content from files (\em{not the options})
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Process some input tensor
		bool	 		process(const Tensor& input);

		/// Change the region of the input tensor to process
		virtual void		setRegion(const TensorRegion& region);

		/// Change the model size (if used with some machine)
		virtual void		setModelSize(const TensorSize& modelSize);

		// Get the ID specific to each spCore
		virtual int		getID() const { return 0; }

		/// Constructs an empty spCore of this kind
		/// (used by <spCoreManager>, this object is automatically deallocated)
		virtual spCore*		getAnInstance() const { return NULL; }

		/// Access the results
		virtual int		getNOutputs() const;
		virtual const Tensor&	getOutput(int index) const;

	protected:

		//////////////////////////////////////////////////////////

		/// Check if the input tensor has the right dimensions and type
		virtual bool		checkInput(const Tensor& input) const = 0;

		/// Allocate (if needed) the output tensors given the input tensor dimensions
		virtual bool		allocateOutput(const Tensor& input) = 0;

		/// Process some input tensor (the input is checked, the outputs are allocated)
		virtual bool		processInput(const Tensor& input) = 0;

		//////////////////////////////////////////////////////////

		/// Delete allocated output tensors
		void			cleanup();

	protected:

		//////////////////////////////////////////////////////////
		/// Attributes

		// Region where to process the input tensor
		TensorRegion		m_region;

		// Model size (if used by a fixed size machine)
		TensorSize		m_modelSize;

		// Processed output tensors
		Tensor**		m_output;
		int			m_n_outputs;
	};

	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::spCoreManager:
	//	Keeps track of the IDs of each spCore.
	//	Given some ID (e.g. found in a model file), it's possible to retrieve a
	//		spCore associated with this ID.
	//
	//	NB: the <spCore>s are deallocated from the spCoreManager.
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class spCoreManager
	{
	public:

		// Destructor
		~spCoreManager();

		// Access to the single instance of this object
		static spCoreManager&	getInstance()
		{
			static spCoreManager manager;
			return manager;
		}

		// Register a new <spCore> with a given ID (supposed to be unique)
		bool			add(spCore* core, const char* name);

		// Get a copy of the <spCore> (empty, no parameters set) for the given ID
		// (returns NULL/0 if the <id> is invalid)
		// The new spCore is automatically deallocated!
		spCore*			get(int id) const;

		// Get the generic name for the given id
		// (returns NULL/0 if the <id> is invalid)
		const char*             getName(int id) const;

		// Get the number of registered spCores
		int                     getNoSpCores() const { return m_size; }

	private:

		// Constructor
		spCoreManager();

		// Deallocate memory
		void			deallocate();

		// Resize the IDs to fit the new <increment>
		void			resize(int increment);

		// Returns the spcores's index with the given ID (or -1, if not found)
		int     		find(int id) const;

		///////////////////////////////////////////////////////////////
		// Attributes

		// spCore prototypes for each ID
		spCore**		m_spcores;
		char**                  m_names;        // Generic name for each spcore
		int			m_size;		// Number of IDs actually used
		int			m_capacity;	// Number of IDs allocated
	};
}

#endif
