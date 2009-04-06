#ifndef SPCORE_INC
#define SPCORE_INC

#include "Object.h"
#include "Tensor.h"
#include "File.h"

namespace Torch
{
	class Tensor;

	class spCore : public Object
	{
	public:
		/// Constructor
		spCore();

		/// Destructor
		virtual ~spCore();

		/// Loading/Saving the content from files (\emph{not the options})
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Process some input tensor
		bool	 		process(const Tensor& input);

		/// Change the region of the input tensor to process
		void			setRegion(const TensorRegion& region);

		/// Change the model size (if used with some machine)
		void			setModelSize(const TensorSize& modelSize);

		/// Access the results
		int			getNOutputs() const;
		const Tensor&		getOutput(int index) const;

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

}

#endif
