#ifndef FRONTAL_FACE_64x40_INC
#define FRONTAL_FACE_64x40_INC

#include "FaceModel.h"

namespace Torch {

/** This class is designed to extract faces of size 64x40 given eye center coordinates

    @author Sebastien Marcel (marcel@idiap.ch)
    \Date
    @version 2.0
    @since 1.0
*/
class FrontalFace64x40 : public FaceModel
{
   	//
	double D_EYES;

public:
	///
	FrontalFace64x40(int postype_);

	/// compute the bounding box from the landmarks
	virtual bool ldm2bbx();

	/// compute the landmarks from the bounding box
	virtual bool bbx2ldm();

	/// compute landmarks #ldm_points_# from the given bounding box #bbx_points_#
	virtual bool bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_);

	///
	virtual ~FrontalFace64x40();
};

}

#endif

