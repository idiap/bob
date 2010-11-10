# Tries to find a local version of Matlab installed
# Andre Anjos - 28.oct.2010

set(MATLAB_FOUND 0)

# This will just calculate what is the best matlab root dir
if (MATLABDIR)
  if (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(MATLAB_ROOT "${MATLABDIR}/bin/glnx86")
  else (CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(MATLAB_ROOT "${MATLABDIR}/bin/glnxa64")
  endif (CMAKE_SIZEOF_VOID_P EQUAL 4)

  find_library(MATLAB_MEX_LIBRARY mex ${MATLAB_ROOT})
  find_library(MATLAB_MX_LIBRARY mx ${MATLAB_ROOT})
  find_library(MATLAB_ENG_LIBRARY eng ${MATLAB_ROOT})
  find_path(MATLAB_INCLUDE_DIR "mex.h" "${MATLABDIR}/extern/include/")

  set(MATLAB_LIBRARIES ${MATLAB_MEX_LIBRARY} ${MATLAB_MX_LIBRARY} ${MATLAB_ENG_LIBRARY})

  if(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)
    set(MATLAB_FOUND 1)
    message( STATUS "Matlab FOUND at ${MATLABDIR}: Enabling extensions...")
    set(MATLAB_VERSION "7.5" CACHE INTERNAL "matlab")
    set(MATLAB_INSTALL_DIRECTORY ${CMAKE_INSTALL_PREFIX}/lib/matlab${MATLAB_VERSION} CACHE INTERNAL "matlab")
  endif(MATLAB_INCLUDE_DIR AND MATLAB_LIBRARIES)

  mark_as_advanced(MATLAB_LIBRARIES
                   MATLAB_MEX_LIBRARY
                   MATLAB_MX_LIBRARY
                   MATLAB_ENG_LIBRARY
                   MATLAB_INCLUDE_DIR
                   MATLAB_FOUND
                   MATLAB_ROOT)

endif (MATLABDIR)
