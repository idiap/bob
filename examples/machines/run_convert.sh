#!/bin/ksh

in_dir="../data/models_torch3/facedetection/frontal/"
out_dir="../data/models/facedetection/frontal/"
models="mct4-2-5-10-50-allface.cascade  mct4-2-5-10-50.cascade mct4.cascade mct5-2-5-10-50-200.cascade"

for model in $models
do
	./`uname -s`_`uname -m`/convert3to5models $in_dir"/"$model $out_dir"/"$model
done

