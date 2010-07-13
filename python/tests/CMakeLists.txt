# Builds all python code in one shot
# Andre Anjos - 13.july.2010

project(pytorch_tests)
cmake_minimum_required(VERSION 2.6)

add_test(python-video-tests ${CMAKE_CURRENT_SOURCE_DIR}/test_video.py)
add_test(python-facefinder-tests ${CMAKE_CURRENT_SOURCE_DIR}/test_facefinder.py)
