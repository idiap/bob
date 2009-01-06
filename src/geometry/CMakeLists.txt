#
# Add the source files to the list of files to be added to the library
#
SET(	geometry_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/Affine2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Circle2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Line2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Matrix2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Octon2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Plane.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Point2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Point3D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Polygon2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Ray2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Rectangle2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/RotationMatrix2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Segment2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Triangle2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Vector2D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Vector3D.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/trigonometry.cc"
	CACHE INTERNAL "sources")

# Include this directory
SET(	geometry_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
