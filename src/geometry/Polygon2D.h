#ifndef POLYGON2D_INC
#define POLYGON2D_INC

#include "Point2D.h"

namespace Torch {

/** This class is designed to handle a polygon in 2D

    @author Sebastien Marcel (marcel@idiap.ch)
    @version 2.0
    \Date
    @since 1.0
*/
class Polygon2D : public geomObject
{
	bool is_allocated;

public:
   	/// number of vertex points
   	int n;

	/// array of points with V[n]=V[0], V[n+1]=V[1]
	Point2D *V;

	//-----

	/** @name constructors */
	//@{
	///
	Polygon2D() {};

	///
	Polygon2D(int n_);

	///
	Polygon2D(int n_, Point2D *V_);
	//@}

	/// destructor
	virtual ~Polygon2D();

	//-----

	/// save the polygon
	virtual void saveFile(File *file);

	/// load a polygon
	virtual void loadFile(File *file);

	/// return the polygon into a string
	virtual const char *sprint();

	//-----

	/// draw the polygon on an image
	virtual void draw(Image *image_, Color color_);

	//---

	/** test the inclusion of a point into the polygon

	    This method uses the winding number test.
	    You can find the original algorithm \URL[here]{http://geometryalgorithms.com/Archive/algorithm_0103/algorithm_0103.htm}.

       	    @param P_ is a point,
            @return the winding number (=0 only if #P_# is outside #V#)

    	    @author Dan Sunday (http://www.softsurfer.com/)
 	*/
	int wn_PnPoly(Point2D P_);

};

}

#endif
