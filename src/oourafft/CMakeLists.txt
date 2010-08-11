project(oourafft)

cmake_minimum_required(VERSION 2.6)

set(src src/alloc.c src/fftsg.c src/fftsg2d.c src/shrtdct.c src/ooura.cc)

include(../macros.cmake)
torch_library(oourafft "${src}" "" "")
