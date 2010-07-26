# Builds all examples in one shot
# Andre Anjos - 28.june.2010

project(examples)
cmake_minimum_required(VERSION 2.6)

# A few handles to build all examples in one shot.
add_custom_target(all-examples)
add_custom_target(create-install-dir COMMAND mkdir -p ${INSTALL_DIR}/bin)
add_custom_target(install-all-examples)

add_subdirectory(boosting)
add_subdirectory(dataset)
add_subdirectory(distribution)
add_subdirectory(facetools)
add_subdirectory(gradient)
add_subdirectory(images)
add_subdirectory(ip)
add_subdirectory(machines)
add_subdirectory(options)
add_subdirectory(scanning)
add_subdirectory(tensors)
add_subdirectory(video)
