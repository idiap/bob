# Finds and configures cblas if it exists on the system. 
# Andre Anjos - 02.july.2010

set(cblas_LIBDIRS /usr/lib;/usr/lib/atlas CACHE INTERNAL "libdirs")
find_path(cblas_INCLUDE NAMES cblas.h)
find_path(lapack_INCLUDE NAMES clapack.h)
find_library(lapack_LIBRARY NAMES clapack lapack PATHS ${cblas_LIBDIRS})
find_library(cblas_LIBRARY NAMES cblas)
find_library(atlas_LIBRARY NAMES atlas)
set(cblas_LIBRARIES ${lapack_LIBRARY};${cblas_LIBRARY};${atlas_LIBRARY} CACHE
    INTERNAL "libraries")
set(cblas_ARCHIVES "" CACHE INTERNAL "archives")
foreach(a ${cblas_LIBRARIES})
	STRING(REGEX REPLACE "(.+)\\.(so|dylib)$" "\\1.a" tmp "${a}")
  set(cblas_ARCHIVES ${cblas_ARCHIVES};${tmp} CACHE INTERNAL "archives")
endforeach(a ${cblas_LIBRARIES})
