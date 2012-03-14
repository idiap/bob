# Finds and configures cblas if it exists on the system. 
# Andre Anjos - 02.july.2010
# Laurent El Shafey - Mar 14 2012

# We start by defining lapack_FOUND to false, let's see what we get next...
set(lapack_FOUND "NO" CACHE INTERNAL "package")

set(suffixes sse2)
find_library(lapack_LIBRARY NAMES lapack PATH_SUFFIXES ${suffixes}) 

if(lapack_LIBRARY)
  set(lapack_FOUND "YES" CACHE INTERNAL "package")
else(lapack_LIBRARY)
  # This will say why we have got to the conclusion to not have found "lapack"
  set(lapack_FOUND "NO")
  message("--   lapack LIBRARY not found!")
endif(lapack_LIBRARY)
