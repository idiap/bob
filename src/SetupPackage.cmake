# This just does some iterative work common to all subpackages of torch

set(${THIS}_INCLUDE ${THIS} CACHE INTERNAL "includes")

set(TORCH5SPRO_INCLUDE ${TORCH5SPRO_INCLUDE};${CMAKE_CURRENT_SOURCE_DIR} CACHE INTERNAL "includes")

foreach(src ${${THIS}_SRC})
  set(TORCH5SPRO_SRC ${TORCH5SPRO_SRC};${CMAKE_CURRENT_SOURCE_DIR}/${src} CACHE INTERNAL "sources")
endforeach(src ${${THIS}_SRC})

foreach(subdir ${${THIS}_SUBDIRS})
  add_subdirectory(${subdir})
endforeach(subdir ${${THIS}_SUBDIRS})
