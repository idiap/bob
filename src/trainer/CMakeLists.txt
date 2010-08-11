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
   "src/Trainer.cc"
   "src/GTFile.cc"
   "src/GradientTrainer.cc"
   "src/StochasticGradientTrainer.cc"
   "src/EMTrainer.cc")

SET(src_gtfiles
   "src/bancaGTFile.cc"
   "src/bbx2eye19x19deye10GTFile.cc"
   "src/cootesGTFile.cc"
   "src/eyecenterGTFile.cc"
   "src/eyecornerGTFile.cc"
   "src/frontalEyeNoseChinGTFile.cc"
   "src/halfprofileEyeNoseChinGTFile.cc"
   "src/profileEyeNoseChinGTFile.cc")

SET(src_boosting
   "src/BoostingRoundLBPTrainer.cc"
   "src/BoostingRoundTrainer.cc"
   "src/BoostingTrainer.cc"
   "src/CascadeTrainer.cc"
   "src/FTrainer.cc"
   "src/LBPRoundTrainer.cc"
   "src/StumpTrainer.cc"
   "src/WeakLearner.cc")

set(src 
	"${src_trainer}"
	"${src_gtfiles}"
	"${src_boosting}")

include(../macros.cmake)
torch_library(trainer "${src}" "${deps}" "${shared}")
