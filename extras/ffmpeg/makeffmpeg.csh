#!/bin/csh -f

# Building FFmpeg library
echo "Building FFmpeg library ..."

\rm -rf build_`uname -s`_`uname -m`
\rm -rf include

cd ffmpeg
./configure --prefix=`pwd`/../build_`uname -s`_`uname -m` --disable-zlib --disable-bzlib --disable-ffmpeg --disable-ffserver --disable-ffplay --enable-gpl --enable-shared
make
make install
cd ..
mv build_`uname -s`_`uname -m`/include .

echo "FFmpeg library built."

