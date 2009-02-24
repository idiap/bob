#ifndef _TORCHVISION_SCANNING_PROFILE_DATA_SET_H_
#define _TORCHVISION_SCANNING_PROFILE_DATA_SET_H_

#include "DataSet.h"		// <ProfileDataSet> is a <DataSet>
#include "Sample.h"

namespace Torch
{
namespace Profile
{
       /////////////////////////////////////////////////////////////////////////
	// Torch::Profile::ProfileDataSet:
	//	- implementation of the DataSet over some Profile::Distribution
	//		and some profile feature
	//	- returns 1D DoubleTensor of the size given by the profile feature size
	//		(check Sample.h header for FeatureSizes[])
	//
	//	NB: the targets will be automatically assigned to 1x1 DoubleTensors (-1, +1)
	//		using the given profile distribution!
	//		=> <setTarget> won't do anything!
	//
	//	NB: the example is buffered, so don't stored it, it will be overwritten!
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ProfileDataSet : public DataSet
	{
	public:

		// Constructor
		ProfileDataSet(const Profile::Distribution& pf_distr, int pf_feature);

		// Destructor
		virtual ~ProfileDataSet();

		// Access examples
		virtual Tensor* 	getExample(long index);
		virtual Tensor&		operator()(long index);

		// Access targets
		virtual Tensor* 	getTarget(long index);
		virtual void		setTarget(long index, Tensor* index);

		/////////////////////////////////////////////////////////////////

	private:

                /////////////////////////////////////////////////////////////////
		// Attributes

		const Profile::Distribution&	m_distribution;
		int			m_feature;	// Feature to extract from the distribution

		DoubleTensor		m_example;	// Current buffered example
		double*			m_pexample;	// Fast access to example

		DoubleTensor		m_target_pos;	// Targets: positive and negative
		DoubleTensor		m_target_neg;
	};
}
}

#endif
