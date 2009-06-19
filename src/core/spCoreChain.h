#ifndef SPCORE_CHAIN_INC
#define SPCORE_CHAIN_INC

#include "spCore.h"

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::spCoreChain:
	//      - process some tensor given a list of spCores
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class spCoreChain : public spCore
	{
	public:
		/// Constructor
		spCoreChain();

		/// Destructor
		virtual ~spCoreChain();

		/// Change the region of the input tensor to process
		virtual void		setRegion(const TensorRegion& region);

		/// Change the model size (if used with some machine)
		virtual void		setModelSize(const TensorSize& modelSize);

		/// Manage the chain of <spCore> to use
		void			clear();
		bool			add(spCore* core);

		/// Access the results
		int			getNOutputs() const;
		const Tensor&		getOutput(int index) const;

		int			getNCores() const { return m_n_cores; }

	protected:

		/// Check if the input tensor has the right dimensions and type
		virtual bool		checkInput(const Tensor& input) const;

		/// Allocate (if needed) the output tensors given the input tensor dimensions
		virtual bool		allocateOutput(const Tensor& input);

		/// Process some input tensor (the input is checked, the outputs are allocated)
		virtual bool		processInput(const Tensor& input);

		//////////////////////////////////////////////////////////
		/// Attributes

		spCore**		m_cores;
		int			m_n_cores;
	};
}

#endif
