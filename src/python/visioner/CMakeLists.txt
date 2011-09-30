project(visioner-python)
cmake_minimum_required(VERSION 2.6)

set(pysrc
   "src/localize.cc"
   "src/main.cc"
   )

include(../../cmake/macros.cmake)

torch_python_bindings(visioner "${pysrc}" "core" "torch_visioner")

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

target_link_libraries(pytorch_visioner pytorch_core_utils)
add_dependencies(pytorch_visioner python-core_header_install visioner)
torch_python_install(visioner)
  
# Our test units
torch_python_add_test(python-visioner-detection-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_detection.py)
torch_python_add_test(python-visioner-localization-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_localization.py)
torch_python_add_test(python-visioner-facebox-script-1
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facebox.py dummy --self-test=1)
torch_python_add_test(python-visioner-facebox-script-2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facebox.py dummy --self-test=2)
torch_python_add_test(python-visioner-facepoints-script-1
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facepoints.py dummy --self-test=1)
torch_python_add_test(python-visioner-facepoints-script-2
  ${CMAKE_CURRENT_SOURCE_DIR}/script/visioner_facepoints.py dummy --self-test=2)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
