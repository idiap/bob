project(oourafft)

cmake_minimum_required(VERSION 2.4)

set(src alloc.c fftsg.c fftsg2d.c shrtdct.c)

include(../macros.cmake)
torch_package(oourafft "${src}" "" "")
