#ifndef FRONTAL_FACE_19x19_INC
#define FRONTAL_FACE_19x19_INC

#include "FaceModel.h"

namespace Torch {

/** This class is designed to extract faces of size 19x19 given eye center coordinates

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Yann Rodriguez (rodrig@idiap.ch)
    \Date
    @version 2.0
    @since 1.0
*/
class FrontalFace19x19 : public FaceModel
{
	//
	double D_EYES;

	//
	double Y_UPPER;

public:
	///
	FrontalFace19x19(int postype_);

	/// compute the bounding box from the landmarks
	virtual bool ldm2bbx();

	/// compute the landmarks from the bounding box
	virtual bool bbx2ldm();

	/// compute landmarks #ldm_points_# from the given bounding box #bbx_points_#
	virtual bool bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_);

	///
	virtual ~FrontalFace19x19();
};

}

#endif

