#ifndef FACE_MODEL_INC
#define FACE_MODEL_INC

#include "ShapeModel.h"
#include "vision.h"

namespace Torch {

/* Face and Head Anthropometry constants

   see "Anthropometry of the Head and Face" L.G. Farkas, Raven Press
*/
#define PUPIL_SE 33.4  // pupil-facial middle distance  (pp 275)
#define ZY_ZY 139.1    // width of the face  (pp 253)
#define EN_GN 117.7    // lower half of the craniofacial height  (pp 257)
#define TR_GN 187.2    // physiognomical height of the face  (pp 254)

#define OR_SCI 38.0    // Combined height of the orbit and the eyebrow  (pp 278)

#define EX_EX 91.2     // biocular width  (pp 272)

#define PUPIL_OR 12.6  // pupil-lower lid height  (pp 281)

#define AL_AL 34.9     // width of the nose  (pp 286)
#define N_SN 54.8      // height of the nose  (pp 288)

#define CH_CH 54.4     // width of the mouth  (pp 306)
#define SN_STO 22.3    // upper lip height  (pp 304)
#define STO_SL 19.7    // lower lip width  (pp 306)

#define SN_LS 14.85    // height of the cutaneous upper lip (pp 304)
#define SN_GN 68.45    // height of the lower face (pp 255)
#define G_SN 67.2      // distance between the glabella and the subnasale (pp 257)

/** This class is designed to model shape and bounding box of a face in an image

    The bounding box is a rectangle.

    @author Sebastien Marcel (marcel@idiap.ch)
    @author Yann Rodriguez (yann.rodriguez@idiap.ch)
    \Date
    @version 2.0
    @since 1.0
*/
class FaceModel : public ShapeModel
{
protected:
	/** @name type of pos file

	    \begin{verbatim}
	       1 = eye centers (IDIAP format)
	       2 = BANCA format
	       3 = eye corners
	       4 = eye corners + nose tip + chin
	       5 = left eye corners + right eye center + nose tip + chin
	       6 = left eye center + nose tip + chin
	       68-71 = Tim Cootes formats
	    \end{verbatim}
	*/
	int postype;

public:
	/** @name index of specific facial features stored in landmarks
	*/
	//@{
	/// right eye
	int r_eye_idx;

	/// left eye
	int l_eye_idx;

	/// nose tip
	int nosetip_idx;

	/// chin
	int chin_idx;
	//@}

	/** @name size of the face model

	    Example: 15x20, 19x19, 64x80, ..
	*/
	//@{
	///
	double model_width;

	///
	double model_height;
	//@}

   	///
	sPoint2D bbx_center;

	///
	int bbx_width;

	///
	int bbx_height;

	//-----

	/// creates a face model
	FaceModel(int postype_);

	/// set the type of pos file format
	void setPostype(int postype_) { postype = postype_; };

	/// get the (in-plane) angle
	virtual double getAngle();

	/// get the center of rotation
        virtual sPoint2D getCenter();

	/// get coordinates of the right eye if exists, return (0,0) otherwise
	sPoint2D getReye();

	/// get coordinates of the left eye if exists, return (0,0) otherwise
	sPoint2D getLeye();

	/// get coordinates of the nose tip if exists, return (0,0) otherwise
	sPoint2D getNosetip();

	/// get coordinates of the chin if exists, return (0,0) otherwise
	sPoint2D getChin();

	/// loads facial features of one face in a pos file according to the postype
	virtual void loadFile(File *file);

	///
	virtual ~FaceModel();
};

}

#endif

