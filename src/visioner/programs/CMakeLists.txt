project(visioner_programs)

include (${QT_USE_FILE})
set(incdir ${cxx_incdir};${QT_INCLUDE_DIR})

# Define compilation options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()

# Visioner applications
bob_add_executable(bob_visioner classifier "classifier.cc")
bob_add_executable(bob_visioner classifier_eval "classifier_eval.cc")
bob_add_executable(bob_visioner detector "detector.cc")
bob_add_executable(bob_visioner detector2bbx "detector2bbx.cc")
bob_add_executable(bob_visioner detector_eval "detector_eval.cc")
bob_add_executable(bob_visioner downscaler "downscaler.cc")
bob_add_executable(bob_visioner drawlbps "drawlbps.cc")
bob_add_executable(bob_visioner drawmb_ctf "drawmb_ctf.cc")
bob_add_executable(bob_visioner drawmbs "drawmbs.cc")
bob_add_executable(bob_visioner face2bbx "face2bbx.cc")
bob_add_executable(bob_visioner feature_stats "feature_stats.cc")
bob_add_executable(bob_visioner gt2pts "gt2pts.cc")
bob_add_executable(bob_visioner localizer "localizer.cc")
bob_add_executable(bob_visioner localizer_eval "localizer_eval.cc")
bob_add_executable(bob_visioner localizer_eval_ex "localizer_eval_ex.cc")
bob_add_executable(bob_visioner max_threads "max_threads.cc")
bob_add_executable(bob_visioner model_stats "model_stats.cc")
bob_add_executable(bob_visioner param2model "param2model.cc")
bob_add_executable(bob_visioner pts2gt "pts2gt.cc")
bob_add_executable(bob_visioner readcmuprofile "readcmuprofile.cc")
bob_add_executable(bob_visioner readmit+cmu "readmit+cmu.cc")
bob_add_executable(bob_visioner readmnist "readmnist.cc")
bob_add_executable(bob_visioner trainer "trainer.cc")

# The graphical 'vgui' app
set(vgui_src
    "vgui/fmap_item.cc"
    "vgui/fmap_scene.cc"
    "vgui/fmap_widget.cc"
    "vgui/settings.cc"
    "vgui/settings_dialog.cc"
    "vgui/image_collection.cc"
    "vgui/main.cc"
    "vgui/controls.cc"
    "vgui/extended_table.cc"
    "vgui/extended_item_delegate.cc"
    )

set(vgui_mocs
    "vgui/fmap_scene.h"
    "vgui/fmap_widget.h"
    "vgui/extended_table.h"
    "vgui/extended_item_delegate.h"
    )
  
#set (vgui_uis
#     "vgui/mainwindow.ui"
#    )

set(vgui_resrc
    "vgui/icons.qrc"
    )


# This command will generate rules that will run rcc on all files from
# SAMPLE_RCS in result SAMPLE_RC_SRCS variable will contain paths to files
# produced by rcc
qt4_add_resources(vgui_resrc "${vgui_resrc}")

# This will run uic on .ui files: only run when required to avoid modifying the
# file (f.ex. the date it was generated on)
qt4_wrap_ui(vgui_uis "${vgui_uis}")

# And finally this will run moc:
qt4_wrap_cpp(vgui_mocs "${vgui_mocs}")

bob_add_executable(bob_visioner vgui "${vgui_src};${vgui_mocs};${vgui_uis};${vgui_resrc}")
