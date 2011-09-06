project(core-python)
cmake_minimum_required(VERSION 2.6)

# Python bindings
set(pysrc
   "src/version.cc"
   "src/exception.cc"
   "src/logging.cc"
   "src/object.cc"
   "src/tensor.cc"
   "src/profile.cc"
   "src/array_check.cc"
   "src/repmat.cc"
   "src/reshape.cc"
   "src/main.cc"
   )

set(pysrc_array
   "src/TypeMapper.cc"
   "src/ndarray.cc"
   "src/tinyvector.cc"
   "src/range.cc"
   )

macro(mktmpl fin fout par1 par2 par3)
  # adds the command to generate the source file
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${fout} 
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/../../bin/mktmpl.py -t "${par1}" -t "${par2}" -t "${par3}" "${CMAKE_CURRENT_SOURCE_DIR}/src/${fin}" "${CMAKE_CURRENT_BINARY_DIR}/${fout}"
    COMMENT "Generating Python/C++ binding for ${fout}...")

  # appends the source file in the list of dependencies for the target
  list(APPEND pysrc_array "${CMAKE_CURRENT_BINARY_DIR}/${fout}")
endmacro(mktmpl fin fout par1 par2 par3)

# iterates over each type/num-of-dimensions that need iteration
foreach(d 1 2 3 4)
  mktmpl(array_math.tcc array_math_bool_${d}.cc "%mathfun%=bool" "%type%=bool" "%dim%=${d}")
  mktmpl(array_arith.tcc array_arith_bool_${d}.cc "%mathfun%=bool" "%type%=bool" "%dim%=${d}")
  foreach(t int8 int16 int32 int64)
    mktmpl(array_math.tcc array_math_${t}_${d}.cc "%mathfun%=int" "%type%=${t}" "%dim%=${d}")
    mktmpl(array_arith.tcc array_arith_${t}_${d}.cc "%mathfun%=int" "%type%=${t}" "%dim%=${d}")
  endforeach(t int8 int16 int32 int64)
  foreach(t uint8 uint16 uint32 uint64)
    mktmpl(array_math.tcc array_math_${t}_${d}.cc "%mathfun%=int" "%type%=${t}" "%dim%=${d}")
    mktmpl(array_arith.tcc array_arith_${t}_${d}.cc "%mathfun%=uint" "%type%=${t}" "%dim%=${d}")
  endforeach(t uint8 uint16 uint32 uint64)
  foreach(t float32 float64)
    mktmpl(array_math.tcc array_math_${t}_${d}.cc "%mathfun%=float" "%type%=${t}" "%dim%=${d}")
    mktmpl(array_arith.tcc array_arith_${t}_${d}.cc "%mathfun%=float" "%type%=${t}" "%dim%=${d}")
  endforeach(t float32 float64)
  foreach(t complex64 complex128)
    mktmpl(array_math.tcc array_math_${t}_${d}.cc "%mathfun%=complex" "%type%=${t}" "%dim%=${d}")
    mktmpl(array_arith.tcc array_arith_${t}_${d}.cc "%mathfun%=complex" "%type%=${t}" "%dim%=${d}")
  endforeach(t complex64 complex128)
endforeach(d 1 2 3 4)

list(APPEND pysrc_array
   "src/array_base.cc"
   "src/array_constructors.cc"
   "src/array_indexing.cc"
   "src/array_indexing_1.cc"
   "src/array_indexing_2.cc"
   "src/array_indexing_3.cc"
   "src/array_indexing_4.cc"
   "src/array_info.cc"
   "src/array_reduction_1.cc"
   "src/array_reduction_2.cc"
   "src/array_reduction_3.cc"
   "src/array_reduction_4.cc"
   "src/array_memory.cc"
   "src/array_order.cc"
   "src/array_cast.cc"
   "src/array_convert.cc"
   "src/array_storage.cc"
   "src/array_as_row.cc"
   "src/array_cat.cc"
   "src/main_array.cc"
   )

set(pysrc_vector
   "src/vector.cc"
   "src/array_vector_1.cc"
   "src/array_vector_2.cc"
   "src/array_vector_3.cc"
   "src/array_vector_4.cc"
   "src/main_vector.cc"
   )

set(pysrc_mapstring
   "src/mapstring.cc"
   "src/arraymapstring.cc"
   "src/main_mapstring.cc"
   )

set(pysrc_tuple
   "src/tuple.cc"
   "src/main_tuple.cc"
   )

set(pysrc_random
   "src/random.cc"
   "src/main_random.cc"
   )

include(../../cmake/macros.cmake)

torch_python_package_bindings(core "${pysrc}")
torch_header_install(core-array-python core "torch::python::core")
torch_python_submodule_bindings(core array "${pysrc_array}")
add_dependencies(pytorch_core_array core-array-python_header_install)
torch_python_submodule_bindings(core vector "${pysrc_vector}")
torch_python_submodule_bindings(core mapstring "${pysrc_mapstring}")
torch_python_submodule_bindings(core tuple "${pysrc_tuple}")
torch_python_submodule_bindings(core random "${pysrc_random}")
torch_python_install(core)

# Our test units
torch_python_add_test(python-core-array-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array.py)
torch_python_add_test(python-core-array-concatenation-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array_cat.py)
torch_python_add_test(python-core-array-math-tests
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_array_math.py)
torch_python_add_test(python-core-exception-tests 
  ${CMAKE_CURRENT_SOURCE_DIR}/test/test_exception.py)

# Installs all python scripts
file(COPY script/ DESTINATION ${CMAKE_INSTALL_PREFIX}/bin FILES_MATCHING PATTERN "*.py")
