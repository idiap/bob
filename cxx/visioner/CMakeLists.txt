PROJECT(visioner-cxx)
cmake_minimum_required(VERSION 2.6)

# This defines the dependencies of this package
set(deps "lbfgs;core") #other bob subprojects

include (${QT_USE_FILE})
include_directories(SYSTEM ${QT_INCLUDE_DIR})
set(shared "${QT_LIBRARIES};${Boost_IOSTREAMS_LIBRARY_RELEASE};${Boost_SERIALIZATION_LIBRARY_RELEASE};${Boost_THREAD_LIBRARY_RELEASE}")

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
    "src/threads.cc"
    "src/util.cc"
    "src/vision.cc"
)

bob_library(visioner "${src}" "${deps}" "${shared}")

set(progshared "${Boost_PROGRAM_OPTIONS_LIBRARY_RELEASE}")

# Visioner applications
bob_add_executable(visioner classifier "programs/classifier.cc" "${progshared}")
bob_add_executable(visioner classifier_eval "programs/classifier_eval.cc" "${progshared}")
bob_add_executable(visioner detector "programs/detector.cc" "${progshared}")
bob_add_executable(visioner detector2bbx "programs/detector2bbx.cc" "${progshared}")
bob_add_executable(visioner detector_eval "programs/detector_eval.cc" "${progshared}")
bob_add_executable(visioner downscaler "programs/downscaler.cc" "${progshared}")
bob_add_executable(visioner drawlbps "programs/drawlbps.cc" "${progshared}")
bob_add_executable(visioner drawmb_ctf "programs/drawmb_ctf.cc" "${progshared}")
bob_add_executable(visioner drawmbs "programs/drawmbs.cc" "${progshared}")
bob_add_executable(visioner face2bbx "programs/face2bbx.cc" "${progshared}")
bob_add_executable(visioner feature_stats "programs/feature_stats.cc" "${progshared}")
bob_add_executable(visioner gt2pts "programs/gt2pts.cc" "${progshared}")
bob_add_executable(visioner localizer "programs/localizer.cc" "${progshared}")
bob_add_executable(visioner localizer_eval "programs/localizer_eval.cc" "${progshared}")
bob_add_executable(visioner localizer_eval_ex "programs/localizer_eval_ex.cc" "${progshared}")
bob_add_executable(visioner max_threads "programs/max_threads.cc" "${progshared}")
bob_add_executable(visioner model_stats "programs/model_stats.cc" "${progshared}")
bob_add_executable(visioner param2model "programs/param2model.cc" "${progshared}")
bob_add_executable(visioner pts2gt "programs/pts2gt.cc" "${progshared}")
bob_add_executable(visioner readcmuprofile "programs/readcmuprofile.cc" "${progshared}")
bob_add_executable(visioner readmit+cmu "programs/readmit+cmu.cc" "${progshared}")
bob_add_executable(visioner readmnist "programs/readmnist.cc" "${progshared}")
bob_add_executable(visioner trainer "programs/trainer.cc" "${progshared}")

# The graphical 'vgui' app
set(vgui_src
    "programs/vgui/fmap_item.cc"
    "programs/vgui/fmap_scene.cc"
    "programs/vgui/fmap_widget.cc"
    "programs/vgui/settings.cc"
    "programs/vgui/settings_dialog.cc"
    "programs/vgui/image_collection.cc"
    "programs/vgui/main.cc"
    "programs/vgui/controls.cc"
    "programs/vgui/extended_table.cc"
    "programs/vgui/extended_item_delegate.cc"
    )

set(vgui_mocs
    "programs/vgui/fmap_scene.h"
    "programs/vgui/fmap_widget.h"
    "programs/vgui/extended_table.h"
    "programs/vgui/extended_item_delegate.h"
    )
  
#set (vgui_uis
#     "programs/vgui/mainwindow.ui"
#    )

set(vgui_resrc
    "programs/vgui/icons.qrc"
    )


# This command will generate rules that will run rcc on all files from
# SAMPLE_RCS in result SAMPLE_RC_SRCS variable will contain paths to files
# produced by rcc
QT4_ADD_RESOURCES(vgui_resrc "${vgui_resrc}")

# This will run uic on .ui files: only run when required to avoid modifying the
# file (f.ex. the date it was generated on)
QT4_WRAP_UI(vgui_uis "${vgui_uis}")

# And finally this will run moc:
QT4_WRAP_CPP(vgui_mocs "${vgui_mocs}")

# We need this to be able to include headers produced by uic in our code
# (CMAKE_BINARY_DIR holds a path to the build directory, while
# INCLUDE_DIRECTORIES() works just like INCLUDEPATH from qmake)
include_directories(${CMAKE_BINARY_DIR})

bob_add_executable(visioner vgui "${vgui_src};${vgui_mocs};${vgui_uis};${vgui_resrc}" "${progshared}")
