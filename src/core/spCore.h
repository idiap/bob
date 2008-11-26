#ifndef SPCORE_INC
#define SPCORE_INC

#include "Object.h"

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

		/// Process some input tensor
		bool	 		process(const Tensor& input);

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

		Tensor**		m_output;
		int			m_n_outputs;
	};

}

#endif
