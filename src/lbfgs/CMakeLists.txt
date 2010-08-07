project(lbfgs)

cmake_minimum_required(VERSION 2.4)

set(src lbfgs.c)

include(../macros.cmake)
torch_package(lbfgs "${src}" "" "")
