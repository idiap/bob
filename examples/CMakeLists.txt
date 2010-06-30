# Builds all examples in one shot
# Andre Anjos - 28.june.2010

project(examples)
cmake_minimum_required(VERSION 2.6)

add_custom_target(build-examples COMMENT "Building all examples...")
add_custom_target(install-examples COMMENT "Installing all examples...")

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
