project(lbfgs)

cmake_minimum_required(VERSION 2.6)

set(src src/lbfgs.c)

include(../macros.cmake)
torch_library(lbfgs "${src}" "" "")
