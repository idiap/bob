#!/bin/csh -f

# Cleaning FFmpeg library
echo "Cleaning FFmpeg library ..."

cd ffmpeg
make clean
make distclean
#\rm -rf ../install/*
cd ..

