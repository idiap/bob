#ifndef _TORCHVISION_SCANNING_CONTEXT_DATA_SET_H_
#define _TORCHVISION_SCANNING_CONTEXT_DATA_SET_H_

#include "core/DataSet.h"		// <ContextDataSet> is a <DataSet>
#include "scanning/Context.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ContextDataSet:
	//	- implementation of the DataSet over context features
	//	- returns 1D DoubleTensor of the size given by the context feature size
	//		(check Context.h header for FeatureSizes[])
	//
	//	NB: the targets will be automatically assigned to 1x1 DoubleTensors (0, +1)
	//		using the given context distribution!
	//		=> <setTarget> won't do anything!
	//
	//	NB: the example is buffered, so don't stored it, it will be overwritten!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ContextDataSet : public Torch::DataSet
	{
	public:

		// Constructor
		ContextDataSet(int pf_feature = 0);

		// Destructor
		virtual ~ContextDataSet();

		// Access examples - overriden
		virtual Tensor* 	getExample(long index);
		virtual Tensor&		operator()(long index);

		// Access targets - overriden
		virtual Tensor* 	getTarget(long index);
		virtual void		setTarget(long index, Tensor* target);

		// Reset to a new context feature
		void			reset(int ctx_feature);

		// Distribution manipulation
		void			clear();
		void			cumulate(bool positive, const Context& context);
		void			cumulate(const Context& gt_context);
		const Context*		getContext(long index) const;
		bool			isPosContext(long index) const;

		// Save the distributions
		bool			save(const char* dir_data, const char* name) const;

		// Load the distributions
		bool			load(const char* dir_data, const char* name);

	private:

		// Save some distribution
		bool			save(const char* basename, unsigned char mask) const;

		// Resize some distribution to fit new samples
		static Context**	resize(Context** old_data, long capacity, long increment);
		static unsigned char*	resize(unsigned char* old_data, long capacity, long increment);

		// Delete stored contexts
		void			cleanup();

		enum Mask
		{
			Positive = 0x00,
			Negative,
			GroundTruth
		};

		/////////////////////////////////////////////////////////////////
		// Attributes

		// Feature to extract from the distribution
		int			m_feature;

		// Context distribution
		Context**		m_contexts;		// [No. examples] x [No. features]
		unsigned char*		m_masks;		// Negative, positive or ground truth
		long			m_capacity;		// Allocated contexts

		// Targets: positive and negative
		DoubleTensor		m_target_neg;
		DoubleTensor		m_target_pos;
	};
}

#endif
