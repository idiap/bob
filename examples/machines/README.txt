#########################################################
# Conversion of the torch3vision models to the torch5spro
#########################################################
./Linux_i686/convert3to5models ../data/models_torch3/facedetection/frontal/mct4-2-5-10-50-allface.cascade ../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade

./Linux_i686/convert3to5models ../data/models_torch3/facedetection/frontal/mct4-2-5-10-50.cascade ../data/models/facedetection/frontal/mct4-2-5-10-50.cascade

./Linux_i686/convert3to5models ../data/models_torch3/facedetection/frontal/mct4.cascade ../data/models/facedetection/frontal/mct4.cascade

./Linux_i686/convert3to5models ../data/models_torch3/facedetection/frontal/mct5-2-5-10-50-200.cascade ../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade

#########################################################
# Testing the converted models on 19x19 bindatas
#########################################################
./Linux_i686/test19x19models ../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade /idiap/common_vision/visidiap/databases/bindata/faces/bindata19x19/cmu/cmu.bindata

./Linux_i686/test19x19models ../data/models/facedetection/frontal/mct4-2-5-10-50.cascade /idiap/common_vision/visidiap/databases/bindata/faces/bindata19x19/cmu/cmu.bindata

./Linux_i686/test19x19models ../data/models/facedetection/frontal/mct4.cascade /idiap/common_vision/visidiap/databases/bindata/faces/bindata19x19/cmu/cmu.bindata

./Linux_i686/test19x19models ../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade /idiap/common_vision/visidiap/databases/bindata/faces/bindata19x19/cmu/cmu.bindata



