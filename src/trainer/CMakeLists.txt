PROJECT(trainer)
cmake_minimum_required(VERSION 2.4)

# This defines the dependencies of this package
set(deps core ip machine scanning lbfgs) #other torch subprojects

# Avoids warning during linkage configuration see: cmake --help-policy CMP0003
# for details.
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

# ffmpeg, somehow, cannot be found by scanning
if(ffmpeg_FOUND)
  link_directories(${ffmpeg_LIBRARY_DIRS})
endif(ffmpeg_FOUND)

SET(src_trainer
	"GTFile.cc"
	"GradientTrainer.cc"
	"StochasticGradientTrainer.cc"
	"EMTrainer.cc")

SET(src_gtfiles
	"bancaGTFile.cc"
	"bbx2eye19x19deye10GTFile.cc"
	"cootesGTFile.cc"
	"eyecenterGTFile.cc"
	"eyecornerGTFile.cc"
	"frontalEyeNoseChinGTFile.cc"
	"halfprofileEyeNoseChinGTFile.cc"
	"profileEyeNoseChinGTFile.cc")

SET(src_boosting
	"BoostingRoundLBPTrainer.cc"
	"BoostingRoundTrainer.cc"
	"BoostingTrainer.cc"
	"CascadeTrainer.cc"
	"FTrainer.cc"
	"LBPRoundTrainer.cc"
	"StumpTrainer.cc"
	"WeakLearner.cc")

set(src 
	"${src_trainer}"
	"${src_gtfiles}"
	"${src_boosting}")

include(../macros.cmake)
torch_package(trainer "${src}" "${deps}" "${shared}")
