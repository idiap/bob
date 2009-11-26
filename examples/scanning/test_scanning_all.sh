#!/bin/sh

image_dir="/idiap/common_vision/visidiap/databases/cmu-test/images_gray"

# Scan each image in the directory
for image in `ls $image_dir/*.pgm`
do
	# Multiscale
	time ./`uname -s`_`uname -m`/scanning $image "facefinder.multiscale.params"

	# Pyramid
	time ./`uname -s`_`uname -m`/scanning $image "facefinder.pyramid.params"

	# Context
	time ./`uname -s`_`uname -m`/scanning $image "facefinder.context.params"
done

