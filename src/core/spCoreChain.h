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

		/// Loading/Saving the content from files (\emph{not the options})
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Process some input tensor
		bool	 		process(const Tensor& input);

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

	protected:

		//////////////////////////////////////////////////////////
		/// Attributes

		spCore**		m_cores;
		int			m_n_cores;
	};
}

#endif
