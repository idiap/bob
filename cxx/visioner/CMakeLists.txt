PROJECT(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "lbfgs;core") #other bob subprojects

include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDE_DIR})
set(shared "${QT_LIBRARIES};${Boost_IOSTREAMS_LIBRARY_RELEASE};${Boost_SERIALIZATION_LIBRARY_RELEASE}")

# Enables OpenMP if found
if (OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${OpenMP_CXX_FLAGS}")
endif ()

# This defines the list of source files inside this package.
set(src
    "src/averager.cc"
    "src/cv_classifier.cc"
    "src/cv_detector.cc"
    "src/cv_draw.cc"
    "src/cv_localizer.cc"
    "src/dataset.cc"
    "src/diag_exp_loss.cc"
    "src/diag_log_loss.cc"
    "src/diag_loss.cc"
    "src/diag_symexp_loss.cc"
    "src/diag_symlog_loss.cc"
    "src/histogram.cc"
    "src/image.cc"
    "src/ipyramid.cc"
    "src/jesorsky_loss.cc"
    "src/lut_problem.cc"
    "src/lut_problem_ept.cc"
    "src/lut_problem_var.cc"
    "src/mdecoder.cc"
    "src/ml.cc"
    "src/model.cc"
    "src/object.cc"
    "src/param.cc"
    "src/sampler.cc"
    "src/tagger_keypoint_oxy.cc"
    "src/tagger_object.cc"
    "src/taylor_booster.cc"
    "src/util.cc"
    "src/vision.cc"
)

bob_library(visioner "${src}" "${deps}" "${shared}")

# Visioner applications
bob_add_executable(visioner classifier programs/classifier.cc)
bob_add_executable(visioner classifier_eval programs/classifier_eval.cc)
bob_add_executable(visioner detector programs/detector.cc)
bob_add_executable(visioner detector2bbx programs/detector2bbx.cc)
bob_add_executable(visioner detector_eval programs/detector_eval.cc)
bob_add_executable(visioner downscaler programs/downscaler.cc)
bob_add_executable(visioner drawlbps programs/drawlbps.cc)
bob_add_executable(visioner drawmb_ctf programs/drawmb_ctf.cc)
bob_add_executable(visioner drawmbs programs/drawmbs.cc)
bob_add_executable(visioner face2bbx programs/face2bbx.cc)
bob_add_executable(visioner feature_stats programs/feature_stats.cc)
bob_add_executable(visioner gt2pts programs/gt2pts.cc)
bob_add_executable(visioner localizer programs/localizer.cc)
bob_add_executable(visioner localizer_eval programs/localizer_eval.cc)
bob_add_executable(visioner localizer_eval_ex programs/localizer_eval_ex.cc)
bob_add_executable(visioner max_threads programs/max_threads.cc)
bob_add_executable(visioner model_stats programs/model_stats.cc)
bob_add_executable(visioner param2model programs/param2model.cc)
bob_add_executable(visioner pts2gt programs/pts2gt.cc)
bob_add_executable(visioner readcmuprofile programs/readcmuprofile.cc)
bob_add_executable(visioner readmit+cmu programs/readmit+cmu.cc)
bob_add_executable(visioner readmnist programs/readmnist.cc)
bob_add_executable(visioner trainer programs/trainer.cc)
