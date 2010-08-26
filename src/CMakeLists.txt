# Configures and builds torch
# Andre Anjos - 13.august.2010

project(torch5spro)
cmake_minimum_required(VERSION 2.6)

# Set the compiler flags and build type(release or debug)
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -O3 -g -Wall -ffast-math -fomit-frame-pointer -msse2")
set(CMAKE_CXX_FLAGS_RELEASE "-pthread -g -Wall")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
set(CMAKE_CXX_FLAGS_DEBUG "-pthread -g -Wall")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")

# This is where we install cmake files for external projects
set(cmakedir share/cmake)

# Sets up externals and install
include(cmake/torch-external.cmake)
foreach(ext ${torch_DEPENDENCIES}) 
  find_package(${ext} REQUIRED PATHS external)
  install(FILES external/${ext}Config.cmake DESTINATION ${cmakedir})
endforeach(ext ${torch_DEPENDENCIES}) 
install(FILES cmake/torch-external.cmake DESTINATION ${cmakedir})

# Enables the testing framework
enable_testing()

# This makes sure we have all headers within reach before starting to compile
include_directories(${CMAKE_INSTALL_PREFIX}/include/torch)

add_subdirectory(python)
add_subdirectory(oourafft)
add_subdirectory(lbfgs)
add_subdirectory(th)
add_subdirectory(core)
add_subdirectory(sp)
add_subdirectory(ap)
add_subdirectory(ip)
add_subdirectory(measurer)
add_subdirectory(machine)
add_subdirectory(scanning)
add_subdirectory(trainer)

install(EXPORT torch DESTINATION ${cmakedir})
install(FILES cmake/torchConfig.cmake DESTINATION ${cmakedir})
