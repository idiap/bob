# Tries to find a local version of ImageMagick++ installed
# Andre Anjos - 22.february.2011

include(FindPkgConfig)

pkg_check_modules(ImageMagick REQUIRED ImageMagick++>=6.5)

if(ImageMagick_FOUND)
  set(ImageMagick_INCLUDE_DIRS ${ImageMagick_INCLUDE_DIRS} CACHE INTERNAL "incdirs")
  include_directories(SYSTEM ${ImageMagick_INCLUDE_DIRS})
  link_directories(${ImageMagick_LIBRARY_DIRS})
  add_definitions("-DHAVE_IMAGE_MAGICK_PP=1")
endif(ImageMagick_FOUND)
