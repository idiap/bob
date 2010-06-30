#
# Add the source files to the list of files to be added to the library
#
PROJECT(geometry)
set(THIS geometry)

SET(${THIS}_SRC
		"Affine2D.cc"
		"Circle2D.cc"
		"Line2D.cc"
		"Matrix2D.cc"
		"Octon2D.cc"
		"Plane.cc"
		"Point2D.cc"
		"Point3D.cc"
		"Polygon2D.cc"
		"Ray2D.cc"
		"Rectangle2D.cc"
		"RotationMatrix2D.cc"
		"Segment2D.cc"
		"Triangle2D.cc"
		"Vector2D.cc"
		"Vector3D.cc"
		"trigonometry.cc"
	CACHE INTERNAL "sources")

include(../SetupPackage.cmake)
