PROJECT(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "lbfgs;core") #other bob subprojects

include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDE_DIR})
set(shared "${QT_LIBRARIES};${Boost_IOSTREAMS_LIBRARY_RELEASE};${Boost_SERIALIZATION_LIBRARY_RELEASE}")

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
