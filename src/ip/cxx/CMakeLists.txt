PROJECT(bob_ip)

# This defines the dependencies of this package
set(bob_deps "bob_io;bob_math;bob_sp")
set(shared "${bob_deps}")
set(incdir ${cxx_incdir})

# This defines the list of source files inside this package.
set(src
   "block.cc"
   "color.cc"
   "Exception.cc"
   "HornAndSchunckFlow.cc"
   "crop.cc"
   "shift.cc"
   "TanTriggs.cc"
   "GeomNorm.cc"
   "maxRectInMask.cc"
   "FaceEyesNorm.cc"
   "GaborWaveletTransform.cc"
   "histo.cc"
   "BlockCellGradientDescriptors.cc"
   "HOG.cc"
   "LBP.cc"
   "LBP4R.cc"
   "LBP8R.cc"
   "LBP16R.cc"
   "LBPTop.cc"
   "GLCM.cc"
   "GLCMProp.cc"
   "Sobel.cc"
   "Gaussian.cc"
   "WeightedGaussian.cc"
   "MultiscaleRetinex.cc"
   "SelfQuotientImage.cc"
   "SpatioTemporalGradient.cc"
   "DCTFeatures.cc"
   "GaussianScaleSpace.cc"
   "SIFT.cc"
)

# If we have VLFeat installed, enable the compilation of relevant modules
if(WITH_VLFEAT)
  list(APPEND incdir "${VLFEAT_INCLUDE_DIR}")
  list(APPEND shared "${VLFEAT_LIBRARY}")
  list(APPEND src 
    "VLSIFT.cc"
    "VLDSIFT.cc"
    )
endif()

# Define the library, compilation and linkage options
bob_sort_headers(incdir)
foreach(inc ${incdir})
  include_directories(SYSTEM ${inc})
endforeach()
bob_add_library(${PROJECT_NAME} "${src}")
target_link_libraries(${PROJECT_NAME} ${shared})

# Defines tests for this package
bob_add_test(${PROJECT_NAME} block test/block.cc)
bob_add_test(${PROJECT_NAME} crop test/crop.cc)
bob_add_test(${PROJECT_NAME} dctfeatures test/dctfeatures.cc)
bob_add_test(${PROJECT_NAME} extrapolateMask test/extrapolateMask.cc)
bob_add_test(${PROJECT_NAME} flipflop test/flipflop.cc)
bob_add_test(${PROJECT_NAME} gwt test/GaborWaveletTransform.cc)
bob_add_test(${PROJECT_NAME} gaussian test/Gaussian.cc)
bob_add_test(${PROJECT_NAME} gaussianScaleSpace test/GaussianScaleSpace.cc)
bob_add_test(${PROJECT_NAME} weightedGaussian test/WeightedGaussian.cc)
bob_add_test(${PROJECT_NAME} median test/Median.cc)
bob_add_test(${PROJECT_NAME} gammaCorrection test/gammaCorrection.cc)
bob_add_test(${PROJECT_NAME} geomnorm test/geomnorm.cc)
bob_add_test(${PROJECT_NAME} facenorm test/facenorm.cc)
bob_add_test(${PROJECT_NAME} integral test/integral.cc)
bob_add_test(${PROJECT_NAME} lbp test/LBP.cc)
bob_add_test(${PROJECT_NAME} lbphsfeatures test/lbphsfeatures.cc)
bob_add_test(${PROJECT_NAME} maxRectInMask test/maxRectInMask.cc)
bob_add_test(${PROJECT_NAME} msr test/MSR.cc)
bob_add_test(${PROJECT_NAME} rotate test/rotate.cc)
bob_add_test(${PROJECT_NAME} scale test/scale.cc)
bob_add_test(${PROJECT_NAME} shear test/shear.cc)
bob_add_test(${PROJECT_NAME} shift test/shift.cc)
bob_add_test(${PROJECT_NAME} sift test/SIFT.cc)
bob_add_test(${PROJECT_NAME} tantriggs test/TanTriggs.cc)
bob_add_test(${PROJECT_NAME} sobel test/Sobel.cc)
bob_add_test(${PROJECT_NAME} zigzag test/zigzag.cc)

# Pkg-Config generator
bob_pkgconfig(${PROJECT_NAME} "${bob_deps}")
