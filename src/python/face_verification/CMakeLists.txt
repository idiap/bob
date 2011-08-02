project(face_verification-python)
cmake_minimum_required(VERSION 2.6)
 
include(../../cmake/macros.cmake)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")

