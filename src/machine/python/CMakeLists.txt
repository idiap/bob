project(bob_machine_py)

# This defines the dependencies of this package
set(bob_deps "bob_machine;bob_python")
set(shared "${bob_deps}")
set(incdir ${python_INCLUDE_DIRS})

# This defines the list of source files inside this package.
set(src
   "machine.cc"
   "gabor.cc"
   "gaussian.cc"
   "gmm.cc"
   "kmeans.cc"
   "activation.cc"
   "linear.cc"
   "mlp.cc"
   "linearscoring.cc"
   "ztnorm.cc"
   "jfa.cc"
   "wiener.cc"
   "plda.cc"
   "bic.cc"
   "version.cc"
   "main.cc"
   )
 
if(LIBSVM_FOUND)
  list(APPEND incdir ${LIBSVM_INCLUDE_DIR})
  list(APPEND src "svm.cc")
endif()

# Define the library, compilation and linkage options
list(REMOVE_DUPLICATES incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
