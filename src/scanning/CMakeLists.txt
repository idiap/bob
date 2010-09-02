PROJECT(scanning)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps lbfgs core ip measurer machine trainer) #other torch subprojects

add_definitions(-DHAVE_LBFGS)
add_definitions(-DUSE_SSE)

# This defines the list of source files inside this package.
set(src_scanning
   "src/Pattern.cc"
   "src/ipGeomNorm.cc"
   "src/ipSWEvaluator.cc"
   "src/ipSWPruner.cc"
   "src/ScaleExplorer.cc"
   "src/Explorer.cc"
   "src/Scanner.cc"
   "src/Selector.cc"
##   "src/PatternTracker.cc"
   "src/FaceFinder.cc")

SET(src_gtfiles
   "src/GTFile.cc"
   "src/bancaGTFile.cc"
   "src/bbx2eye19x19deye10GTFile.cc"
   "src/cootesGTFile.cc"
   "src/eyecenterGTFile.cc"
   "src/eyecornerGTFile.cc"
   "src/frontalEyeNoseChinGTFile.cc"
   "src/halfprofileEyeNoseChinGTFile.cc"
   "src/profileEyeNoseChinGTFile.cc")

set(src_context
   "src/Context.cc"
   "src/ContextDataSet.cc"
   "src/ContextMachine.cc"
   "src/ContextTrainer.cc"
   "src/LRMachine.cc"
   "src/LRTrainer.cc")

set(src_explorers
   "src/ExhaustiveScaleExplorer.cc"
   "src/SpiralScaleExplorer.cc"
   "src/RandomScaleExplorer.cc"
   "src/MSExplorer.cc"
   "src/ContextExplorer.cc"
   "src/PyramidExplorer.cc"
   "src/TrackContextExplorer.cc")

set(src_pruners
   "src/ipSWVariancePruner.cc")

set(src_selectors
   "src/DummySelector.cc"
   "src/OverlapSelector.cc"
   "src/MeanShiftSelector.cc")

set(src
	"${src_scanning}"
	"${src_gtfiles}"
	"${src_context}"
	"${src_explorers}"
	"${src_pruners}"
	"${src_selectors}")

include(../cmake/macros.cmake)
torch_library(scanning "${src}" "${deps}" "${shared}")

# Python bindings
set(pysrc
   "python/src/explorer.cc"
   "python/src/facefinder.cc"
   "python/src/pattern.cc"
   "python/src/scanner.cc"
   "python/src/ipgeomnorm.cc"
   "python/src/gtfile.cc"
   "python/src/main.cc"
   )
 
torch_python_bindings(scanning "${pysrc}")
torch_python_install(scanning)
