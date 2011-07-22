project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/localize.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_bindings(visioner "${pysrc}" "" "torch_visioner")

# The visioner expects a flat directory structure
get_target_property(save pytorch_visioner COMPILE_FLAGS)
if(NOT save)
  set(save "")
endif(NOT save)
set(baseinc ${CMAKE_INSTALL_PREFIX}/include/torch/visioner)
set_target_properties(pytorch_visioner PROPERTIES COMPILE_FLAGS "${save} -I${baseinc}/vision -I${baseinc}/ml -I${baseinc}/proc -I${baseinc}/util")

# Specially for these bindings, we will have to find libtorch_visioner which
# comes from an external build.
get_target_property(save pytorch_visioner LINK_FLAGS)
if(NOT save)
  set(save "")
endif(NOT save)
set(baselib ${CMAKE_INSTALL_PREFIX}/lib)
set_target_properties(pytorch_visioner PROPERTIES LINK_FLAGS "${save} -L${baselib}")

# Visioner depends on Qt4
include (${QT_USE_FILE})

add_dependencies(pytorch_visioner core-array-python_header_install visioner)
torch_python_install(visioner)
  
# Our test units
#torch_python_add_test(python-visioner-kmeans-tests 
#  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_visioner.py)
