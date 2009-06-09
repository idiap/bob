#ifndef _TORCHVISION_SCANNING_PROFILE_DATA_SET_H_
#define _TORCHVISION_SCANNING_PROFILE_DATA_SET_H_

#include "DataSet.h"		// <ProfileDataSet> is a <DataSet>
#include "Profile.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::ProfileDataSet:
	//	- implementation of the DataSet over some Distribution
	//		and some profile feature
	//	- returns 1D DoubleTensor of the size given by the profile feature size
	//		(check Sample.h header for FeatureSizes[])
	//
	//	NB: the targets will be automatically assigned to 1x1 DoubleTensors (0, +1)
	//		using the given profile distribution!
	//		=> <setTarget> won't do anything!
	//
	//	NB: the example is buffered, so don't stored it, it will be overwritten!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ProfileDataSet : public Torch::DataSet
	{
	public:

		// Constructor
		ProfileDataSet(int pf_feature = 0);

		// Destructor
		virtual ~ProfileDataSet();

		// Access examples - overriden
		virtual Tensor* 	getExample(long index);
		virtual Tensor&		operator()(long index);

		// Access targets - overriden
		virtual Tensor* 	getTarget(long index);
		virtual void		setTarget(long index, Tensor* target);

		// Reset to a new profile feature
		void			reset(int pf_feature);

		// Distribution manipulation
		void			clear();
		void			cumulate(bool positive, const Profile& profile);
		void			cumulate(const Profile& gt_profile);
		const Profile*		getProfile(long index) const;
		bool			isPosProfile(long index) const;

		// Save the distributions
		bool			save(const char* dir_data, const char* name) const;

		// Load the distributions
		bool			load(const char* dir_data, const char* name);

	private:

		// Save some distribution
		bool			save(const char* basename, unsigned char mask) const;

		// Resize some distribution to fit new samples
		static Profile**	resize(Profile** old_data, long capacity, long increment);
		static unsigned char*	resize(unsigned char* old_data, long capacity, long increment);

		// Delete stored profiles
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

		// Profile distribution
		Profile**		m_profiles;		// [No. examples] x [No. features]
		unsigned char*		m_masks;		// Negative, positive or ground truth
		long			m_capacity;		// Allocated profiles

		// Targets: positive and negative
		DoubleTensor		m_target_neg;
		DoubleTensor		m_target_pos;
	};
}

#endif
