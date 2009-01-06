#
# Add the source files to the list of files to be added to the library
#
SET(	matrix_SRC
		"${CMAKE_CURRENT_SOURCE_DIR}/Mat.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Mat_operations.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Perm.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Perm_operations.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/Vec.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_givens.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_hessenberg.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_householder.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_low_level.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_lu_factor.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_solve.cc"
		"${CMAKE_CURRENT_SOURCE_DIR}/mx_sym_eig.cc"
	CACHE INTERNAL "sources")

SET(	matrix_INCLUDE
		${CMAKE_CURRENT_SOURCE_DIR}
	CACHE INTERNAL "includes")

