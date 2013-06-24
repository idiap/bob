project(bob_trainer_py${PYVER})

# This defines the dependencies of this package
set(bob_deps "bob_trainer;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir})

set(src
   "linear.cc"
   "kmeans.cc"
   "gmm.cc"
   "mlpbase.cc"
   "rprop.cc"
   "backprop.cc"
   "jfa.cc"
   "ivector.cc"
   "wiener.cc"
   "empca.cc"
   "plda.cc"
   "bic.cc"
   "llr.cc"
   "whitening.cc"
   "wccn.cc"
   "cost.cc"
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
