project(bob_src)

add_subdirectory(lbfgs)

# Required externals
include(blitz.cmake)
include(boost.cmake)
include(python.cmake)

add_subdirectory(core)
add_subdirectory(python)

add_subdirectory(io)
add_subdirectory(measure)
add_subdirectory(sp)
add_subdirectory(math)
add_subdirectory(ip)

include(libsvm.cmake)
add_subdirectory(machine)
add_subdirectory(trainer)

find_package(Qt4 COMPONENTS QtCore QtGui)

if(QT4_FOUND)
  set(HAVE_QT4 ON CACHE BOOL "Has Qt4 installed")
  add_subdirectory(visioner)
  add_subdirectory(daq)
endif()

set(ENABLED_PACKAGES "${ENABLED_PACKAGES}" PARENT_SCOPE)
