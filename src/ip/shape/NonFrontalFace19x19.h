#ifndef NONFRONTAL_FACE_19x19_INC
#define NONFRONTAL_FACE_19x19_INC

#include "FaceModel.h"

namespace Torch {

/** This class is designed to extractsnon-frontal faces of size 19x19 given eye center coordinates, nosetip and chin

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Tiffany Sauquet
    @author Yann Rodriguez (rodrig@idiap.ch)
    \Date
    @version 2.0
    @since 1.0

*/
class NonFrontalFace19x19 : public FaceModel
{
	double D_EYES;
	double Y_UPPER;
	double EYE_CHIN;

	/** pan angle
		0: frontal face
		1: 22.5 degree
		2: 45 degree
		3: 67.5 degree
		4: 90 degree
	*/
	int pan;
public:

	/** creates a non-frontal face model given a pan angle and a postype

	    @param pan_ is the pan angle and can take any of this value (0: frontal face, 1: 22.5 degree, 2: 45 degree, 3: 67.5 degree, 4: 90 degree)
	    @param postype_ is the file format of annotation files (.pos)
	*/
	NonFrontalFace19x19(int pan_, int postype_);

	/// compute the bounding box from the landmarks
	virtual bool ldm2bbx();

	/// compute the landmarks from the bounding box
	virtual bool bbx2ldm();

	/// compute landmarks #ldm_points_# from the given bounding box #bbx_points_#
	virtual bool bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_);

	/// get the (in-plane) angle
	virtual double getAngle();

	/// get the center of rotation
	virtual sPoint2D getCenter();

	///
	virtual ~NonFrontalFace19x19();
};

}

#endif

