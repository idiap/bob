project(ip-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/ipcore.cc"
   "src/color.cc"
   "src/image.cc"
   "src/lbp.cc"
   "src/filters.cc"
   "src/video.cc"
   "src/videotensor.cc"
   "src/vision.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_bindings(ip "${pysrc}")
torch_python_install(ip)
