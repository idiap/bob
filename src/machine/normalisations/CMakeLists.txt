#
# Add the source files to the list of files to be added to the library
#
PROJECT(normalisations)
set(THIS normalisations)

SET(${THIS}_SRC "MeanVarNorm.cc" CACHE INTERNAL "sources")

include(../../SetupPackage.cmake)
