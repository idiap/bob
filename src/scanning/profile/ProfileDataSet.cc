#include "ProfileDataSet.h"

namespace Torch
{
namespace Profile
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ProfileDataSet::ProfileDataSet(const Profile::Distribution& pf_distr, int pf_feature)
	: 	m_distribution(pf_distr),
		m_feature(pf_feature),
		m_example(Profile::FeatureSizes[pf_feature]),
		m_pexample((double*)m_example.dataW()),
		m_target_pos(1),
		m_target_neg(1)
{
	m_example.fill(0.0);
	m_target_pos.fill(1.0);
	m_target_neg.fill(-1.0);
}

/////////////////////////////////////////////////////////////////////////

}
}
