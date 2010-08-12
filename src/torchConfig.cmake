# Locates and loads all torch exported dependencies
find_file(torch_BUILD_INFO torch.cmake)
include(${torch_BUILD_INFO})

# Defines the includes
get_filename_component(torch_CMAKE_DIR ${torch_BUILD_INFO} PATH)
get_filename_component(torch_SHARE_DIR ${torch_CMAKE_DIR} PATH)
get_filename_component(torch_PREFIX ${torch_SHARE_DIR} PATH)

# Loads all externals
find_package(cblas)
find_package(boost)
find_package(jpeg)
find_package(ffmpeg)
find_package(python)

set(torch_INCLUDE_DIRS ${torch_PREFIX}/include/torch)

message("-- Found Torch: ${torch_PREFIX}")
