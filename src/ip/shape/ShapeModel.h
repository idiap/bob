#ifndef SHAPE_MODEL_INC
#define SHAPE_MODEL_INC

#include "Object.h"

namespace Torch {

	class Image;
	class ipRotate;
	class sPoint2D;

	/** This class is designed to model shape and bounding box of 2D objects in images

	    A shape is for example a face, a car or a pedestrian.
	    A bounding box is for examples a rectangle or a polygon.

	    @author Sebastien Marcel (marcel@idiap.ch)
	    \Date
	    @version 2.0
	    @since 1.0
	*/
	class ShapeModel : public Object
	{
	public:
		/// number of landmarks
		int n_ldm_points;

		/// landmarks of the shape
		sPoint2D *ldm_points;

		/// number of point for the bounding box
		int n_bbx_points;

		/// bounding box of the shape
		sPoint2D *bbx_points;

		//-----

		/// creates a shape model
		ShapeModel(int n_bbx_points_);

		/// compute the bounding box from the landmarks
		virtual bool ldm2bbx() = 0;

		/// compute the landmarks from the bounding box
		virtual bool bbx2ldm() = 0;

		/// compute landmarks #ldm_points_# from the given bounding box #bbx_points_#
		virtual bool bbx2ldm(int n_bbx_points_, sPoint2D *bbx_points_, int n_ldm_points_, sPoint2D *ldm_points_) = 0;

		/// rotate the shape using #rot# operator
		// TODO: needs ipRotate to be implemented!!!
		///virtual void rotate(ipRotate *rot_);

		/// draw landmarks in the given image
		virtual void drawLDM(Image *image_);

		/// draw bounding box in the given image
		virtual void drawBBX(Image *image_);

		///
		virtual ~ShapeModel();
	};
}

#endif

