project(lbfgs-cxx)

cmake_minimum_required(VERSION 2.6)

set(src src/lbfgs.c)

bob_library(lbfgs "${src}" "" "")
