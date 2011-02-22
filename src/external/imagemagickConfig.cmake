# Tries to find a local version of ImageMagick++ installed
# Andre Anjos - 22.february.2011

include(FindPkgConfig)

pkg_check_modules(ImageMagick REQUIRED ImageMagick++>=6.5)
add_definitions("-DHAVE_IMAGE_MAGICK_PP=1")
