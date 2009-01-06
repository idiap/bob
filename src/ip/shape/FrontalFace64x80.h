#ifndef FRONTAL_FACE_64x80_INC
#define FRONTAL_FACE_64x80_INC

#include "FaceModel.h"

namespace Torch {

/** This class is designed to extract faces of size 64x80 given eye center coordinates

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Yann Rodriguez (rodrig@idiap.ch)

    \Date
    @version 2.0
    @since 1.0
*/
class FrontalFace64x80 : public FaceModel
{
   	//
	double D_EYES;

	//
        double Y_UPPER;

	//
        double Y_LOWER;

public:
	///
	FrontalFace64x80(int postype_);

	/// compute the bounding box from the landmarks
	virtual bool ldm2bbx();

	/// compute the landmarks from the bounding box
	virtual bool bbx2ldm();

	/// compute landmarks #ldm_points_# from the given bounding box #bbx_points_#
	virtual bool bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_);

	///
	virtual ~FrontalFace64x80();
};

}

#endif

