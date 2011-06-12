# Finds and configures cblas if it exists on the system. 
# Andre Anjos - 02.july.2010

# We start by defining cblas_FOUND to false, let's see what we get next...
set(cblas_FOUND "NO" CACHE INTERNAL "package")

# Here we do some variable cleanup and adjustments
find_path(cblas_INCLUDE NAMES cblas.h)
find_path(lapack_INCLUDE NAMES clapack.h PATHS "/usr/include/atlas")
set(cblas_INCLUDE_DIRS ${cblas_INCLUDE};${lapack_INCLUDE})
list(REMOVE_DUPLICATES cblas_INCLUDE_DIRS)
set(cblas_INCLUDE_DIRS ${cblas_INCLUDE_DIRS} CACHE INTERNAL "incdirs")
include_directories(SYSTEM ${cblas_INCLUDE_DIRS})

set(suffixes sse2)
find_library(lapack_LIBRARY NAMES lapack PATH_SUFFIXES ${suffixes}) 
find_library(cblas_LIBRARY NAMES cblas PATH_SUFFIXES ${suffixes})
set(cblas_LIBRARIES ${lapack_LIBRARY};${cblas_LIBRARY})
list(REMOVE_DUPLICATES cblas_LIBRARIES)
set(cblas_LIBRARIES ${cblas_LIBRARIES} CACHE INTERNAL "libraries")

if(cblas_INCLUDE AND lapack_INCLUDE AND cblas_LIBRARY AND lapack_LIBRARY)
  set(cblas_FOUND "YES" CACHE INTERNAL "package")
  add_definitions("-DUSE_CBLAS=1")
else(cblas_INCLUDE AND lapack_INCLUDE AND cblas_LIBRARY AND lapack_LIBRARY)
  # This will say why we have got to the conclusion to not have found "cblas"
  set(cblas_FOUND "NO")
  if (NOT cblas_INCLUDE)
    message("--   cblas INCLUDE not found!")
  endif (NOT cblas_INCLUDE)
  if (NOT lapack_INCLUDE)
    message("--   lapack INCLUDE not found!")
  endif (NOT lapack_INCLUDE)
  if (NOT lapack_LIBRARY)
    message("--   lapack LIBRARY not found!")
  endif (NOT lapack_LIBRARY)
  if (NOT cblas_LIBRARY)
    message("--   cblas LIBRARY not found!")
  endif (NOT cblas_LIBRARY)
endif(cblas_INCLUDE AND lapack_INCLUDE AND cblas_LIBRARY AND lapack_LIBRARY)
