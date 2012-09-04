project(bob_ip_py)

# This defines the dependencies of this package
set(bob_deps "bob_ip;bob_python")
set(shared "${bob_deps}")
set(incdir ${py_incdir})

set(src
   "version.cc"
   "color.cc"
   "block.cc"
   "crop_shift.cc"
   "extrapolate_mask.cc"
   "flipflop.cc"
   "gamma_correction.cc"
   "integral.cc"
   "scale.cc"
   "shear.cc"
   "zigzag.cc"
   "flow.cc"
   "DCTFeatures.cc"
   "gaussian.cc"
   "WeightedGaussian.cc"
   "Median.cc"
   "Sobel.cc"
   "GaborWaveletTransform.cc"
   "GeomNorm.cc"
   "FaceEyesNorm.cc"
   "rotate.cc"
   "TanTriggs.cc"
   "histo.cc"
   "LBP.cc"
   "MultiscaleRetinex.cc"
   "SelfQuotientImage.cc"
   "drawing.cc"
   "SpatioTemporalGradient.cc"
   "HOG.cc"
   "main.cc"
   )

# If we have VLFeat installed, enable the compilation of relevant modules
if(VLFEAT_FOUND)
  list(APPEND incdir "${VLFEAT_INCLUDE_DIR}")
  list(APPEND src 
    "vlsift.cc"
    "vldsift.cc"
    )
endif(VLFEAT_FOUND)

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
