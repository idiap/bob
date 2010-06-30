# Configures all extras in one shot. 
# Andre Anjos - 28.june.2010

# Sets up all extras in one go. 
project(extras)
cmake_minimum_required(VERSION 2.8)

include(ExternalProject)

# This will handle the build of ffmpeg as a external project
macro(build_ffmpeg version branch)
  if (${CMAKE_BUILD_TYPE} EQUAL "DEBUG")
    set(FFMPEG_DEBUG "--disable-optimizations")
  endif (${CMAKE_BUILD_TYPE} EQUAL "DEBUG")
  ExternalProject_Add(
    ffmpeg
    PREFIX "ffmpeg-${version}"
    SVN_REPOSITORY svn://svn.mplayerhq.hu/ffmpeg/${branch}
    UPDATE_COMMAND svn update .
    CONFIGURE_COMMAND <SOURCE_DIR>/configure --prefix=<INSTALL_DIR> --incdir=${INCLUDE_DIR} --disable-zlib --disable-bzlib --disable-ffmpeg --disable-ffserver --disable-ffplay --enable-gpl --enable-shared ${FFMPEG_DEBUG}
    INSTALL_DIR ${INSTALL_DIR}
    BUILD_COMMAND $(MAKE) -j${CPU_COUNT}
    INSTALL_COMMAND $(MAKE) -j${CPU_COUNT} install
  )
endmacro(build_ffmpeg)

# This will handle the build of jpeg as a external project
macro(build_cmakeable proj)
  ExternalProject_Add(
    ${proj}
    PREFIX "current"
    URL ${CMAKE_CURRENT_SOURCE_DIR}/${proj} 
    CMAKE_ARGS -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DINSTALL_DIR=${INSTALL_DIR} -DINCLUDE_DIR=${INCLUDE_DIR}
    BUILD_COMMAND $(MAKE) -j${CPU_COUNT}
    INSTALL_COMMAND $(MAKE) -j${CPU_COUNT} install
    )
endmacro(build_cmakeable proj)

build_ffmpeg("0.6" "branches/0.6")
build_cmakeable(jpeg)
build_cmakeable(lbfgs)
build_cmakeable(oourafft)
