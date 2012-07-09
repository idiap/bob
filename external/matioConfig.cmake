# Tries to find a local version of matio installed
# Andre Anjos - 9.july.2012

include(FindPkgConfig)
pkg_check_modules(matio matio)

if(matio_FOUND)
  add_definitions("-DHAVE_MATIO=1")

  include(CheckCSourceCompiles)
  set(CMAKE_REQUIRED_INCLUDES ${matio_INCLUDE_DIR})
  set(CMAKE_REQUIRED_LIBRARIES ${matio_LIBRARIES})
  set(CMAKE_REQUIRED_FLAGS ${matio_LDFLAGS})
  CHECK_C_SOURCE_COMPILES("#include <matio.h> 
    int main() { struct ComplexSplit s; }" HAVE_MATIO_OLD_COMPLEXSPLIT)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_LIBRARIES)
  set(CMAKE_REQUIRED_INCLUDES)

endif()
