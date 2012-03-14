# Finds and configures cblas if it exists on the system. 
# Andre Anjos - 02.july.2010
# Laurent El Shafey - Mar 14 2012

# We start by defining lapack_FOUND to false, let's see what we get next...
set(lapack_FOUND "NO" CACHE INTERNAL "package")
  
set(suffixes sse2)
find_library(lapack_LIBRARY NAMES lapack PATH_SUFFIXES ${suffixes}) 
find_library(blas_LIBRARY NAMES cblas PATH_SUFFIXES ${suffixes})
set(lapack_LIBRARIES ${lapack_LIBRARY};${blas_LIBRARY})
list(REMOVE_DUPLICATES lapack_LIBRARIES)
set(lapack_LIBRARIES ${lapack_LIBRARIES} CACHE INTERNAL "libraries")

if(blas_LIBRARY AND lapack_LIBRARY)
  set(lapack_FOUND "YES" CACHE INTERNAL "package")
else(blas_LIBRARY AND lapack_LIBRARY)
  # This will say why we have got to the conclusion to not have found "lapack"
  set(lapack_FOUND "NO")
  if (NOT lapack_LIBRARY)
    message("--   lapack LIBRARY not found!")
  endif (NOT lapack_LIBRARY)
  if (NOT blas_LIBRARY)
    message("--   blas LIBRARY not found!")
  endif (NOT blas_LIBRARY)
endif(blas_LIBRARY AND lapack_LIBRARY)
