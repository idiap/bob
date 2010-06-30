#
# Add the source files to the list of files to be added to the library
#
PROJECT(ap)
set(THIS ap)

SET(${THIS}_SRC "apMFCC.cc" "apLFCC.cc")
include(../SetupPackage.cmake)
include(../InstallParts.cmake)
