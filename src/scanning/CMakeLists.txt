PROJECT(scanning)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core ip machine) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_definitions(-DHAVE_LBFGS)
include_directories("${INCLUDE_DIR}/lbfgs")
set(shared "-llbfgs")

# This defines the list of source files inside this package.
set(src_scanning
	"Pattern.cc"
	"ipSWEvaluator.cc"
	"ipSWPruner.cc"
	"ScaleExplorer.cc"
	"Explorer.cc"
	"Scanner.cc"
	"Selector.cc")
##		"PatternTracker.cc"
#	"FaceFinder.cc")

set(src_context
	"Context.cc"
	"ContextDataSet.cc"
	"ContextMachine.cc")
#	"ContextTrainer.cc")
#	"LRMachine.cc"
#	"LRTrainer.cc")

set(src_explorers
	"ExhaustiveScaleExplorer.cc"
	"SpiralScaleExplorer.cc"
	"RandomScaleExplorer.cc"
	"MSExplorer.cc"
	"ContextExplorer.cc"
	"PyramidExplorer.cc"
	"TrackContextExplorer.cc")

set(src_pruners
	"ipSWVariancePruner.cc")

set(src_selectors
	"DummySelector.cc"
	"OverlapSelector.cc"
	"MeanShiftSelector.cc")

set(src
	"${src_scanning}")
#	"${src_context}"
#	"${src_explorers}"
#	"${src_pruners}"
#	"${src_selectors}")

include(../macros.cmake)
torch_package(scanning "${src}" "${deps}" "${shared}")
