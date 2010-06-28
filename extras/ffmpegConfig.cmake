# Tries to find a local version of ffmpeg/developement installed
# Andre Anjos - 25.june.2010

include(FindPkgConfig)
pkg_check_modules(ffmpeg REQUIRED libavformat libswscale libavutil)

