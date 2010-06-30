#
# Add the source files to the list of files to be added to the library
#
PROJECT(measurer)
set(THIS measurer)

SET(${THIS}_SRC "measurer.cc"
  CACHE INTERNAL "sources")

include(../InstallParts.cmake)
include(../SetupPackage.cmake)
