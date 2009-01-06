#ifndef FRONTAL_FACE_15x20_INC
#define FRONTAL_FACE_15x20_INC

#include "FaceModel.h"

namespace Torch {

/** This class is designed to extract faces of size 15x20 given eye center coordinates
	
    @author Sebastien Marcel (marcel@idiap.ch)
    \Date
    @version 2.0
    @since 1.0
*/
class FrontalFace15x20 : public FaceModel
{
public:
	///
	FrontalFace15x20(int postype_);

	/// compute the bounding box from the landmarks
	virtual bool ldm2bbx();

	/// compute the landmarks from the bounding box
	virtual bool bbx2ldm();

	/// compute landmarks #ldm_points_# from the given bounding box #bbx_points_#
	virtual bool bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_);

	///
	virtual ~FrontalFace15x20();
};

}

#endif

