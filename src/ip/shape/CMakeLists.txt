#
# Add the source files to the list of files to be added to the library
#
SET(	shape_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/ShapeModel.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FaceModel.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FrontalFace15x20.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FrontalFace19x19.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FrontalFace64x40.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FrontalFace64x56.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/FrontalFace64x80.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/NonFrontalFace19x19.cc"
	CACHE INTERNAL "sources")

# Include this directory
SET(	shape_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")
