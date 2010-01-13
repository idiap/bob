#########################################################
# Conversion of the torch3vision cascade models to the torch5spro
#########################################################

./`uname -s`_`uname -m`/convert3to5models ./models_torch3/frontal/mct4-2-5-10-50.cascade mct4-2-5-10-50.cascade

#./`uname -s`Linux`uname -s`_`uname -m`/convert3to5models ./data/models_torch3/facedetection/frontal/mct4-2-5-10-50-allface.cascade ../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade
#./`uname -s`Linux_`uname -m`/convert3to5models ./models_torch3/frontal/mct4-2-5-10-50.cascade ../data/models/facedetection/frontal/mct4-2-5-10-50.cascade
#./`uname -s`Linux_`uname -m`/convert3to5models ./models_torch3/frontal/mct4.cascade ../data/models/facedetection/frontal/mct4.cascade
#./`uname -s`Linux_`uname -m`/convert3to5models ./models_torch3/frontal/mct5-2-5-10-50-200.cascade ../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade

#########################################################
# Testing the converted models on 19x19 bindatas
#########################################################
./`uname -s`_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct4-2-5-10-50-allface.cascade ../data/tensors/nonfacetest.tensor

./`uname -s`_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct4-2-5-10-50.cascade ../data/tensors/nonfacetest.tensor

./`uname -s`_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct4.cascade ../data/tensors/nonfacetest.tensor

./`uname -s`_`uname -m`/test19x19models ../data/models/facedetection/frontal/mct5-2-5-10-50-200.cascade ../data/tensors/nonfacetest.tensor



#########################################################
# Conversion of the torch3vision pyramid cascade models to the torch5spro
#########################################################

# First convert each individual cascade of the pyramid-cascade
mkdir mv

# copy all the file describing the structure of the pyramid-cascade
cp ./models_torch3/mv/*.* mv/

# convert each cascade (the one copied above will be overwriten with the converted cascade)
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/toplevel.model mv/toplevel.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/inplane-router.model mv/inplane-router.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/outplane-router.model mv/outplane-router.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/inplane-frontal.model mv/inplane-frontal.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/inplane-view-22.model mv/inplane-view-22.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/inplane-view+22.model mv/inplane-view+22.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/inplane-view-45.model mv/inplane-view-45.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/inplane-view+45.model mv/inplane-view+45.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/outplane-frontal.model mv/outplane-frontal.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/outplane-view-22.model mv/outplane-view-22.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/outplane-view+22.model mv/outplane-view+22.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/outplane-view-45.model mv/outplane-view-45.model
./`uname -s`_`uname -m`/convert3to5models ./models_torch3/mv/outplane-view+45.model mv/outplane-view+45.model

# Then do the final convertion to a single file !
./`uname -s`_`uname -m`/convert3pyramidto5tree mv pyramid-cascade.tree

# Test the pyramid-cascade
./`uname -s`_`uname -m`/treetest19x19models ../data/models/facedetection/mv/pyramid-cascade-inplane-outplane-0-22-45.tree ../data/tensors/nonfacetest.tensor

