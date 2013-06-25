project(bob_machine_py${PYVER})

# This defines the dependencies of this package
set(bob_deps "bob_machine;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir})

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
   "ivector.cc"
   "wiener.cc"
   "plda.cc"
   "bic.cc"
   "roll.cc"
   "version.cc"
   "main.cc"
   )
 
if(WITH_LIBSVM)
  list(APPEND incdir ${LIBSVM_INCLUDE_DIR})
  list(APPEND src "svm.cc")
endif()

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
