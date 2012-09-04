project(sp)

find_path(FFTW3_INCLUDE_DIR fftw3.h)
find_library(FFTW3_LIBRARY NAMES fftw3)

set(FFTW3_FOUND FALSE)
if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
  set(FFTW3_FOUND TRUE)
  find_package_message(FFTW3 "Found FFTW3: ${FFTW3_LIBRARY}" "[${FFTW3_LIBRARY}][${FFTW3_INCLUDE_DIR}]")
endif()

add_subdirectory(cxx)
add_subdirectory(python)
 
set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
