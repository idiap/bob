project(lbfgs)

cmake_minimum_required(VERSION 2.6)

set(src src/lbfgs.c)

include(../cmake/macros.cmake)
torch_library(lbfgs "${src}" "" "")
