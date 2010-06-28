project(ffmpeg)

cmake_minimum_required(VERSION 2.6)

macro(checkout branch)
  set(CHECKOUT_DIR "ffmpeg-${branch}")
  string(REGEX REPLACE "/" "-" CHECKOUT_DIR ${CHECKOUT_DIR})
  set(CHECKOUT_DIR "${ffmpeg_SOURCE_DIR}/${CHECKOUT_DIR}")
  string(REGEX REPLACE "\n" "" CHECKOUT_DIR ${CHECKOUT_DIR})
  ## This will either checkout or update your currently checked-out sources
  message("Checking-out/Updating ${CHECKOUT_DIR}...")
  execute_process(COMMAND svn checkout svn://svn.mplayerhq.hu/ffmpeg/${branch} ${CHECKOUT_DIR})
endmacro(checkout)

macro(configure codir installdir incdir)
  ## May need to add "--disable-vdpau" switch if using NVidia card
  message("Configuring ${codir}...")
  execute_process(COMMAND ./configure --prefix=${installdir} --incdir=${incdir} --disable-zlib --disable-bzlib --disable-ffmpeg --disable-ffserver --disable-ffplay --enable-gpl --enable-shared WORKING_DIRECTORY ${codir}) 
endmacro(configure)

macro(build codir)
  message("Compiling and installing ${codir}...")
  execute_process(COMMAND make all install WORKING_DIRECTORY ${codir})
endmacro(build)

macro(build_ffmpeg branch)
  checkout(${branch}) # this will set up CHECKOUT_DIR
  configure(${CHECKOUT_DIR} ${INSTALL_DIR} ${INCLUDE_DIR})
  build(${CHECKOUT_DIR})
endmacro(build_ffmpeg)

if(NOT DEFINED ${INSTALL_DIR})
  # User is trying to execute this one locally, define the directories.
  include(../Directories.cmake)
endif(NOT DEFINED ${INSTALL_DIR})

build_ffmpeg("branches/0.6")
