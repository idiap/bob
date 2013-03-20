PROJECT(bob_visioner)

# This defines the dependencies of this package
set(bob_deps "bob_lbfgs;bob_core")
include (${QT_USE_FILE})
set(incdir ${cxx_incdir};${QT_INCLUDES})
set(shared "${bob_deps};${QT_LIBRARIES};${Boost_IOSTREAMS_LIBRARY_RELEASE};${Boost_SERIALIZATION_LIBRARY_RELEASE};${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}")

# This defines the list of source files inside this package.
set(src
    "averager.cc"
    "cv_classifier.cc"
    "cv_detector.cc"
    "cv_draw.cc"
    "cv_localizer.cc"
    "dataset.cc"
    "diag_exp_loss.cc"
    "diag_log_loss.cc"
    "diag_loss.cc"
    "diag_symexp_loss.cc"
    "diag_symlog_loss.cc"
    "histogram.cc"
    "image.cc"
    "ipyramid.cc"
    "jesorsky_loss.cc"
    "lut_problem.cc"
    "lut_problem_ept.cc"
    "lut_problem_var.cc"
    "mdecoder.cc"
    "ml.cc"
    "model.cc"
    "object.cc"
    "param.cc"
    "sampler.cc"
    "tagger_keypoint_oxy.cc"
    "tagger_object.cc"
    "taylor_booster.cc"
    "threads.cc"
    "util.cc"
    "vision.cc"
)

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
