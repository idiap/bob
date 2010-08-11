project(lbfgs)

cmake_minimum_required(VERSION 2.6)

set(src lbfgs.c)

include(../macros.cmake)
torch_library(th "${src}" "" "")
