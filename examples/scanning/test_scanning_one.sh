#!/bin/sh

image="../data/images/1001_f_g1_s01_1001_en_1.jpeg"

# Multiscale
time ./`uname -s`_`uname -m`/scanning $image "facefinder.multiscale.params"

# Pyramid
time ./`uname -s`_`uname -m`/scanning $image "facefinder.pyramid.params"

# Context
time ./`uname -s`_`uname -m`/scanning $image "facefinder.context.params"
