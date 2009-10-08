#########################################################
# Conversion of the torch3vision models to the torch5spro
#########################################################
./Linux_`uname -m`/convert3to5models ../data/models_torch3/facedetection/frontal/mct4-2-5-10-50-allface.cascade ../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade

./Linux_`uname -m`/convert3to5models ../data/models_torch3/facedetection/frontal/mct4-2-5-10-50.cascade ../data/models/facedetection/frontal/mct4-2-5-10-50.cascade

./Linux_`uname -m`/convert3to5models ../data/models_torch3/facedetection/frontal/mct4.cascade ../data/models/facedetection/frontal/mct4.cascade

./Linux_`uname -m`/convert3to5models ../data/models_torch3/facedetection/frontal/mct5-2-5-10-50-200.cascade ../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade

#########################################################
# Testing the converted models on 19x19 bindatas
#########################################################
./Linux_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade ../data/tensors/nonfacetest.tensor

./Linux_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct4-2-5-10-50.cascade ../data/tensors/nonfacetest.tensor

./Linux_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct4.cascade ../data/tensors/nonfacetest.tensor

./Linux_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade ../data/tensors/nonfacetest.tensor



